from __future__ import annotations
import logging
from functools import cached_property
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, List, Optional
from langchain_core.callbacks import (
from langchain_core.language_models.llms import BaseLLM
from langchain_core.load.serializable import Serializable
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import root_validator
class GigaChat(_BaseGigaChat, BaseLLM):
    """`GigaChat` large language models API.

    To use, you should pass login and password to access GigaChat API or use token.

    Example:
        .. code-block:: python

            from langchain_community.llms import GigaChat
            giga = GigaChat(credentials=..., scope=..., verify_ssl_certs=False)
    """
    payload_role: str = 'user'

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

    def _create_llm_result(self, response: Any) -> LLMResult:
        generations = []
        for res in response.choices:
            finish_reason = res.finish_reason
            gen = Generation(text=res.message.content, generation_info={'finish_reason': finish_reason})
            generations.append([gen])
            if finish_reason != 'stop':
                logger.warning('Giga generation stopped with reason: %s', finish_reason)
            if self.verbose:
                logger.info('Giga response: %s', res.message.content)
        token_usage = response.usage
        llm_output = {'token_usage': token_usage, 'model_name': response.model}
        return LLMResult(generations=generations, llm_output=llm_output)

    def _generate(self, prompts: List[str], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, stream: Optional[bool]=None, **kwargs: Any) -> LLMResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            generation: Optional[GenerationChunk] = None
            stream_iter = self._stream(prompts[0], stop=stop, run_manager=run_manager, **kwargs)
            for chunk in stream_iter:
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            return LLMResult(generations=[[generation]])
        payload = self._build_payload(prompts)
        response = self._client.chat(payload)
        return self._create_llm_result(response)

    async def _agenerate(self, prompts: List[str], stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, stream: Optional[bool]=None, **kwargs: Any) -> LLMResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            generation: Optional[GenerationChunk] = None
            stream_iter = self._astream(prompts[0], stop=stop, run_manager=run_manager, **kwargs)
            async for chunk in stream_iter:
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            return LLMResult(generations=[[generation]])
        payload = self._build_payload(prompts)
        response = await self._client.achat(payload)
        return self._create_llm_result(response)

    def _stream(self, prompt: str, stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Iterator[GenerationChunk]:
        payload = self._build_payload([prompt])
        for chunk in self._client.stream(payload):
            if chunk.choices:
                content = chunk.choices[0].delta.content
                yield GenerationChunk(text=content)
                if run_manager:
                    run_manager.on_llm_new_token(content)

    async def _astream(self, prompt: str, stop: Optional[List[str]]=None, run_manager: Optional[AsyncCallbackManagerForLLMRun]=None, **kwargs: Any) -> AsyncIterator[GenerationChunk]:
        payload = self._build_payload([prompt])
        async for chunk in self._client.astream(payload):
            if chunk.choices:
                content = chunk.choices[0].delta.content
                yield GenerationChunk(text=content)
                if run_manager:
                    await run_manager.on_llm_new_token(content)

    class Config:
        extra = 'allow'