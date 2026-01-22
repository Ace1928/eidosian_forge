import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, cast
import requests
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.llms.utils import enforce_stop_tokens
def _convert_chunk_to_message_message(self, chunk: str) -> AIMessageChunk:
    data = json.loads(chunk.encode('utf-8'))
    return AIMessageChunk(content=data.get('response', ''))