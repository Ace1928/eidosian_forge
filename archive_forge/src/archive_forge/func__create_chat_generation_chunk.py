from __future__ import annotations
import logging
from typing import (
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import (
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import AIMessageChunk, BaseMessage, BaseMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from langchain_community.adapters.openai import (
from langchain_community.chat_models.openai import _convert_delta_to_message_chunk
def _create_chat_generation_chunk(self, data: Mapping[str, Any], default_chunk_class: Type[BaseMessageChunk]) -> Tuple[ChatGenerationChunk, Type[BaseMessageChunk]]:
    chunk = _convert_delta_to_message_chunk({'content': data.get('text', '')}, default_chunk_class)
    finish_reason = data.get('finish_reason')
    generation_info = dict(finish_reason=finish_reason) if finish_reason is not None else None
    default_chunk_class = chunk.__class__
    gen_chunk = ChatGenerationChunk(message=chunk, generation_info=generation_info)
    return (gen_chunk, default_chunk_class)