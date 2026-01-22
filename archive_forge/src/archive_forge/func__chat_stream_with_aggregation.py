import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union, cast
from langchain_core._api import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_community.llms.ollama import OllamaEndpointNotFoundError, _OllamaCommon
def _chat_stream_with_aggregation(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, verbose: bool=False, **kwargs: Any) -> ChatGenerationChunk:
    final_chunk: Optional[ChatGenerationChunk] = None
    for stream_resp in self._create_chat_stream(messages, stop, **kwargs):
        if stream_resp:
            chunk = _chat_stream_response_to_chat_generation_chunk(stream_resp)
            if final_chunk is None:
                final_chunk = chunk
            else:
                final_chunk += chunk
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk, verbose=verbose)
    if final_chunk is None:
        raise ValueError('No data received from Ollama stream.')
    return final_chunk