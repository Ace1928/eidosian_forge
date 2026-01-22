from __future__ import annotations
import importlib
from typing import (
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import (
from langchain_core.pydantic_v1 import BaseModel
from typing_extensions import Literal
def _convert_message_chunk(chunk: BaseMessageChunk, i: int) -> dict:
    _dict: Dict[str, Any] = {}
    if isinstance(chunk, AIMessageChunk):
        if i == 0:
            _dict['role'] = 'assistant'
        if 'function_call' in chunk.additional_kwargs:
            _dict['function_call'] = chunk.additional_kwargs['function_call']
            if i == 0:
                _dict['content'] = None
        else:
            _dict['content'] = chunk.content
    else:
        raise ValueError(f'Got unexpected streaming chunk type: {type(chunk)}')
    if _dict == {'content': ''}:
        _dict = {}
    return _dict