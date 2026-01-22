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
def _truncate_params(payload: Dict[str, Any]) -> None:
    """Truncate temperature and top_p parameters between [0.01, 0.99].

    ZhipuAI only support temperature / top_p between (0, 1) open interval,
    so we truncate them to [0.01, 0.99].
    """
    temperature = payload.get('temperature')
    top_p = payload.get('top_p')
    if temperature is not None:
        payload['temperature'] = max(0.01, min(0.99, temperature))
    if top_p is not None:
        payload['top_p'] = max(0.01, min(0.99, top_p))