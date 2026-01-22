import asyncio
import re
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union, cast
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
class GenericFakeChatModel(BaseChatModel):
    """A generic fake chat model that can be used to test the chat model interface.

    * Chat model should be usable in both sync and async tests
    * Invokes on_llm_new_token to allow for testing of callback related code for new
      tokens.
    * Includes logic to break messages into message chunk to facilitate testing of
      streaming.
    """
    messages: Iterator[Union[AIMessage, str]]
    'Get an iterator over messages.\n\n    This can be expanded to accept other types like Callables / dicts / strings\n    to make the interface more generic if needed.\n\n    Note: if you want to pass a list, you can use `iter` to convert it to an iterator.\n\n    Please note that streaming is not implemented yet. We should try to implement it\n    in the future by delegating to invoke and then breaking the resulting output\n    into message chunks.\n    '

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> ChatResult:
        """Top Level call"""
        message = next(self.messages)
        if isinstance(message, str):
            message_ = AIMessage(content=message)
        else:
            message_ = message
        generation = ChatGeneration(message=message_)
        return ChatResult(generations=[generation])

    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model."""
        chat_result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        if not isinstance(chat_result, ChatResult):
            raise ValueError(f'Expected generate to return a ChatResult, but got {type(chat_result)} instead.')
        message = chat_result.generations[0].message
        if not isinstance(message, AIMessage):
            raise ValueError(f'Expected invoke to return an AIMessage, but got {type(message)} instead.')
        content = message.content
        if content:
            assert isinstance(content, str)
            content_chunks = cast(List[str], re.split('(\\s)', content))
            for token in content_chunks:
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=token, id=message.id))
                if run_manager:
                    run_manager.on_llm_new_token(token, chunk=chunk)
                yield chunk
        if message.additional_kwargs:
            for key, value in message.additional_kwargs.items():
                if key == 'function_call':
                    for fkey, fvalue in value.items():
                        if isinstance(fvalue, str):
                            fvalue_chunks = cast(List[str], re.split('(,)', fvalue))
                            for fvalue_chunk in fvalue_chunks:
                                chunk = ChatGenerationChunk(message=AIMessageChunk(id=message.id, content='', additional_kwargs={'function_call': {fkey: fvalue_chunk}}))
                                if run_manager:
                                    run_manager.on_llm_new_token('', chunk=chunk)
                                yield chunk
                        else:
                            chunk = ChatGenerationChunk(message=AIMessageChunk(id=message.id, content='', additional_kwargs={'function_call': {fkey: fvalue}}))
                            if run_manager:
                                run_manager.on_llm_new_token('', chunk=chunk)
                            yield chunk
                else:
                    chunk = ChatGenerationChunk(message=AIMessageChunk(id=message.id, content='', additional_kwargs={key: value}))
                    if run_manager:
                        run_manager.on_llm_new_token('', chunk=chunk)
                    yield chunk

    @property
    def _llm_type(self) -> str:
        return 'generic-fake-chat-model'