from __future__ import annotations
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterator, List, Optional, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks.manager import (
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_community.utilities.vertexai import (
def _prepare_params(self, stop: Optional[List[str]]=None, stream: bool=False, **kwargs: Any) -> dict:
    stop_sequences = stop or self.stop
    params_mapping = {'n': 'candidate_count'}
    params = {params_mapping.get(k, k): v for k, v in kwargs.items()}
    params = {**self._default_params, 'stop_sequences': stop_sequences, **params}
    if stream or self.streaming:
        params.pop('candidate_count')
    return params