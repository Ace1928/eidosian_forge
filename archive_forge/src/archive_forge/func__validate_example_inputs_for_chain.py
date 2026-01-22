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
def _validate_example_inputs_for_chain(first_example: Example, chain: Chain, input_mapper: Optional[Callable[[Dict], Any]]) -> None:
    """Validate that the example inputs match the chain input keys."""
    if input_mapper:
        first_inputs = input_mapper(first_example.inputs)
        missing_keys = set(chain.input_keys).difference(first_inputs)
        if not isinstance(first_inputs, dict):
            raise InputFormatError(f'When using an input_mapper to prepare dataset example inputs for a chain, the mapped value must be a dictionary.\nGot: {first_inputs} of type {type(first_inputs)}.')
        if missing_keys:
            raise InputFormatError(f'Missing keys after loading example using input_mapper.\nExpected: {chain.input_keys}. Got: {first_inputs.keys()}')
    else:
        first_inputs = first_example.inputs
        missing_keys = set(chain.input_keys).difference(first_inputs)
        if len(first_inputs) == 1 and len(chain.input_keys) == 1:
            pass
        elif missing_keys:
            raise InputFormatError(f'Example inputs missing expected chain input keys. Please provide an input_mapper to convert the example.inputs to a compatible format for the chain you wish to evaluate.Expected: {chain.input_keys}. Got: {first_inputs.keys()}')