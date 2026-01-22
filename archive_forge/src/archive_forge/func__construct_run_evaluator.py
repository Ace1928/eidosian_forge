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
def _construct_run_evaluator(eval_config: Union[smith_eval_config.SINGLE_EVAL_CONFIG_TYPE, smith_eval_config.CUSTOM_EVALUATOR_TYPE], eval_llm: Optional[BaseLanguageModel], run_type: str, data_type: DataType, example_outputs: Optional[List[str]], reference_key: Optional[str], input_key: Optional[str], prediction_key: Optional[str]) -> RunEvaluator:
    if isinstance(eval_config, RunEvaluator):
        return eval_config
    if isinstance(eval_config, (EvaluatorType, str)):
        if not isinstance(eval_config, EvaluatorType):
            eval_config = EvaluatorType(eval_config)
        evaluator_ = load_evaluator(eval_config, llm=eval_llm)
        eval_type_tag = eval_config.value
    elif isinstance(eval_config, smith_eval_config.EvalConfig):
        kwargs = {'llm': eval_llm, **eval_config.get_kwargs()}
        evaluator_ = load_evaluator(eval_config.evaluator_type, **kwargs)
        eval_type_tag = eval_config.evaluator_type.value
        if isinstance(eval_config, smith_eval_config.SingleKeyEvalConfig):
            input_key = eval_config.input_key or input_key
            prediction_key = eval_config.prediction_key or prediction_key
            reference_key = eval_config.reference_key or reference_key
    elif callable(eval_config):
        return run_evaluator_dec(eval_config)
    else:
        raise ValueError(f'Unknown evaluator type: {type(eval_config)}')
    if isinstance(evaluator_, StringEvaluator):
        if evaluator_.requires_reference and reference_key is None:
            raise ValueError(f'Must specify reference_key in smith_eval.RunEvalConfig to use evaluator of type {eval_type_tag} with dataset with multiple output keys: {example_outputs}.')
        run_evaluator = smith_eval.StringRunEvaluatorChain.from_run_and_data_type(evaluator_, run_type, data_type, input_key=input_key, prediction_key=prediction_key, reference_key=reference_key, tags=[eval_type_tag])
    elif isinstance(evaluator_, PairwiseStringEvaluator):
        raise NotImplementedError(f'Run evaluator for {eval_type_tag} is not implemented. PairwiseStringEvaluators compare the outputs of two different models rather than the output of a single model. Did you mean to use a StringEvaluator instead?\nSee: https://python.langchain.com/docs/guides/evaluation/string/')
    else:
        raise NotImplementedError(f'Run evaluator for {eval_type_tag} is not implemented')
    return run_evaluator