import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
from aiohttp import ClientSession
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Extra, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from langchain_community.utilities.requests import Requests
@property
def _api_key(self) -> str:
    if self.edenai_api_key:
        return self.edenai_api_key.get_secret_value()
    return ''