from __future__ import annotations
import asyncio
import functools
import inspect
import json
import logging
import uuid
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
import yaml
from tenacity import (
from langchain_core._api import deprecated
from langchain_core.caches import BaseCache
from langchain_core.callbacks import (
from langchain_core.globals import get_llm_cache
from langchain_core.language_models.base import BaseLanguageModel, LanguageModelInput
from langchain_core.load import dumpd
from langchain_core.messages import (
from langchain_core.outputs import Generation, GenerationChunk, LLMResult, RunInfo
from langchain_core.prompt_values import ChatPromptValue, PromptValue, StringPromptValue
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.runnables import RunnableConfig, ensure_config, get_config_list
from langchain_core.runnables.config import run_in_executor
@staticmethod
def _get_run_ids_list(run_id: Optional[Union[uuid.UUID, List[Optional[uuid.UUID]]]], prompts: list) -> list:
    if run_id is None:
        return [None] * len(prompts)
    if isinstance(run_id, list):
        if len(run_id) != len(prompts):
            raise ValueError(f"Number of manually provided run_id's does not match batch length. {len(run_id)} != {len(prompts)}")
        return run_id
    return [run_id] + [None] * (len(prompts) - 1)