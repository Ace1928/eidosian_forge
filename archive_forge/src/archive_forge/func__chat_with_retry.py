from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, cast
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import (
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from tenacity import (
@retry_decorator
def _chat_with_retry(**kwargs: Any) -> Any:
    return llm.client.chat(**kwargs)