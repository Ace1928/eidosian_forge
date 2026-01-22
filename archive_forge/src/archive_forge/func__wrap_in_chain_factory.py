from __future__ import annotations
import concurrent.futures
import dataclasses
import functools
import inspect
import logging
import uuid
from datetime import datetime, timezone
from typing import (
from langchain_core._api import warn_deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, messages_from_dict
from langchain_core.outputs import ChatResult, LLMResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.runnables import config as runnable_config
from langchain_core.runnables import utils as runnable_utils
from langchain_core.tracers.evaluation import (
from langchain_core.tracers.langchain import LangChainTracer
from langsmith.client import Client
from langsmith.env import get_git_info, get_langchain_env_var_metadata
from langsmith.evaluation import (
from langsmith.evaluation import (
from langsmith.run_helpers import as_runnable, is_traceable_function
from langsmith.schemas import Dataset, DataType, Example, Run, TracerSession
from langsmith.utils import LangSmithError
from requests import HTTPError
from typing_extensions import TypedDict
from langchain.callbacks.manager import Callbacks
from langchain.chains.base import Chain
from langchain.evaluation.loading import load_evaluator
from langchain.evaluation.schema import (
from langchain.smith import evaluation as smith_eval
from langchain.smith.evaluation import config as smith_eval_config
from langchain.smith.evaluation import name_generation, progress
def _wrap_in_chain_factory(llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY, dataset_name: str='<my_dataset>') -> MCF:
    """Forgive the user if they pass in a chain without memory instead of a chain
    factory. It's a common mistake. Raise a more helpful error message as well."""
    if isinstance(llm_or_chain_factory, Chain):
        chain = llm_or_chain_factory
        chain_class = chain.__class__.__name__
        if llm_or_chain_factory.memory is not None:
            memory_class = chain.memory.__class__.__name__
            raise ValueError(f'Cannot directly evaluate a chain with stateful memory. To evaluate this chain, pass in a chain constructor that initializes fresh memory each time it is called.  This will safegaurd against information leakage between dataset examples.\nFor example:\n\ndef chain_constructor():\n    new_memory = {memory_class}(...)\n    return {chain_class}(memory=new_memory, ...)\n\nrun_on_dataset("{dataset_name}", chain_constructor, ...)')
        return lambda: chain
    elif isinstance(llm_or_chain_factory, BaseLanguageModel):
        return llm_or_chain_factory
    elif isinstance(llm_or_chain_factory, Runnable):
        lcf = llm_or_chain_factory
        return lambda: lcf
    elif callable(llm_or_chain_factory):
        if is_traceable_function(llm_or_chain_factory):
            runnable_ = as_runnable(cast(Callable, llm_or_chain_factory))
            return lambda: runnable_
        try:
            _model = llm_or_chain_factory()
        except TypeError:
            user_func = cast(Callable, llm_or_chain_factory)
            sig = inspect.signature(user_func)
            logger.info(f'Wrapping function {sig} as RunnableLambda.')
            wrapped = RunnableLambda(user_func)
            return lambda: wrapped
        constructor = cast(Callable, llm_or_chain_factory)
        if isinstance(_model, BaseLanguageModel):
            return _model
        elif is_traceable_function(cast(Callable, _model)):
            runnable_ = as_runnable(cast(Callable, _model))
            return lambda: runnable_
        elif not isinstance(_model, Runnable):
            return lambda: RunnableLambda(constructor)
        else:
            return constructor
    return llm_or_chain_factory