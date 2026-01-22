from __future__ import annotations
import asyncio
import inspect
import uuid
import warnings
from abc import ABC, abstractmethod
from typing import (
from langchain_core._api import deprecated
from langchain_core.caches import BaseCache
from langchain_core.callbacks import (
from langchain_core.globals import get_llm_cache
from langchain_core.language_models.base import BaseLanguageModel, LanguageModelInput
from langchain_core.load import dumpd, dumps
from langchain_core.messages import (
from langchain_core.outputs import (
from langchain_core.prompt_values import ChatPromptValue, PromptValue, StringPromptValue
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.runnables.config import ensure_config, run_in_executor
from langchain_core.tracers.log_stream import LogStreamCallbackHandler
def _generate_with_cache(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> ChatResult:
    if isinstance(self.cache, BaseCache):
        llm_cache = self.cache
    else:
        llm_cache = get_llm_cache()
    check_cache = self.cache or self.cache is None
    if check_cache:
        if llm_cache:
            llm_string = self._get_llm_string(stop=stop, **kwargs)
            prompt = dumps(messages)
            cache_val = llm_cache.lookup(prompt, llm_string)
            if isinstance(cache_val, list):
                return ChatResult(generations=cache_val)
        elif self.cache is None:
            pass
        else:
            raise ValueError('Asked to cache, but no cache found at `langchain.cache`.')
    if type(self)._stream != BaseChatModel._stream and kwargs.pop('stream', next((True for h in run_manager.handlers if isinstance(h, LogStreamCallbackHandler)), False) if run_manager else False):
        chunks: List[ChatGenerationChunk] = []
        for chunk in self._stream(messages, stop=stop, **kwargs):
            chunk.message.response_metadata = _gen_info_and_msg_metadata(chunk)
            if run_manager:
                if chunk.message.id is None:
                    chunk.message.id = f'run-{run_manager.run_id}'
                run_manager.on_llm_new_token(cast(str, chunk.message.content), chunk=chunk)
            chunks.append(chunk)
        result = generate_from_stream(iter(chunks))
    elif inspect.signature(self._generate).parameters.get('run_manager'):
        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
    else:
        result = self._generate(messages, stop=stop, **kwargs)
    for idx, generation in enumerate(result.generations):
        if run_manager and generation.message.id is None:
            generation.message.id = f'run-{run_manager.run_id}-{idx}'
        generation.message.response_metadata = _gen_info_and_msg_metadata(generation)
    if len(result.generations) == 1 and result.llm_output is not None:
        result.generations[0].message.response_metadata = {**result.llm_output, **result.generations[0].message.response_metadata}
    if check_cache and llm_cache:
        llm_cache.update(prompt, llm_string, result.generations)
    return result