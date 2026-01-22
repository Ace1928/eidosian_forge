from __future__ import annotations
import logging
from functools import cached_property
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, List, Optional
from langchain_core.callbacks import (
from langchain_core.language_models.llms import BaseLLM
from langchain_core.load.serializable import Serializable
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import root_validator
def _build_payload(self, messages: List[str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {'messages': [{'role': self.payload_role, 'content': m} for m in messages]}
    if self.model:
        payload['model'] = self.model
    if self.profanity_check is not None:
        payload['profanity_check'] = self.profanity_check
    if self.temperature is not None:
        payload['temperature'] = self.temperature
    if self.top_p is not None:
        payload['top_p'] = self.top_p
    if self.max_tokens is not None:
        payload['max_tokens'] = self.max_tokens
    if self.repetition_penalty is not None:
        payload['repetition_penalty'] = self.repetition_penalty
    if self.update_interval is not None:
        payload['update_interval'] = self.update_interval
    if self.verbose:
        logger.info('Giga request: %s', payload)
    return payload