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
def _get_prompt(inputs: Dict[str, Any]) -> str:
    """Get prompt from inputs.

    Args:
        inputs: The input dictionary.

    Returns:
        A string prompt.
    Raises:
        InputFormatError: If the input format is invalid.
    """
    if not inputs:
        raise InputFormatError('Inputs should not be empty.')
    prompts = []
    if 'prompt' in inputs:
        if not isinstance(inputs['prompt'], str):
            raise InputFormatError(f"Expected string for 'prompt', got {type(inputs['prompt']).__name__}")
        prompts = [inputs['prompt']]
    elif 'prompts' in inputs:
        if not isinstance(inputs['prompts'], list) or not all((isinstance(i, str) for i in inputs['prompts'])):
            raise InputFormatError(f"Expected list of strings for 'prompts', got {type(inputs['prompts']).__name__}")
        prompts = inputs['prompts']
    elif len(inputs) == 1:
        prompt_ = next(iter(inputs.values()))
        if isinstance(prompt_, str):
            prompts = [prompt_]
        elif isinstance(prompt_, list) and all((isinstance(i, str) for i in prompt_)):
            prompts = prompt_
        else:
            raise InputFormatError(f'LLM Run expects string prompt input. Got {inputs}')
    else:
        raise InputFormatError(f"LLM Run expects 'prompt' or 'prompts' in inputs. Got {inputs}")
    if len(prompts) == 1:
        return prompts[0]
    else:
        raise InputFormatError(f'LLM Run expects single prompt input. Got {len(prompts)} prompts.')