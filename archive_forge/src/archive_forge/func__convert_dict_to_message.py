from __future__ import annotations
import json
import logging
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
def _convert_dict_to_message(dct: Dict[str, Any]) -> BaseMessage:
    role = dct.get('role')
    content = dct.get('content', '')
    if role == 'system':
        return SystemMessage(content=content)
    if role == 'user':
        return HumanMessage(content=content)
    if role == 'assistant':
        additional_kwargs = {}
        tool_calls = dct.get('tool_calls', None)
        if tool_calls is not None:
            additional_kwargs['tool_calls'] = tool_calls
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    return ChatMessage(role=role, content=content)