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
def _get_jwt_token(api_key: str) -> str:
    """Gets JWT token for ZhipuAI API, see 'https://open.bigmodel.cn/dev/api#nosdk'.

    Args:
        api_key: The API key for ZhipuAI API.

    Returns:
        The JWT token.
    """
    import jwt
    try:
        id, secret = api_key.split('.')
    except ValueError as err:
        raise ValueError(f'Invalid API key: {api_key}') from err
    payload = {'api_key': id, 'exp': int(round(time.time() * 1000)) + API_TOKEN_TTL_SECONDS * 1000, 'timestamp': int(round(time.time() * 1000))}
    return jwt.encode(payload, secret, algorithm='HS256', headers={'alg': 'HS256', 'sign_type': 'SIGN'})