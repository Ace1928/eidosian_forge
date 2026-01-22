from __future__ import annotations
import base64
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union, cast
from urllib.parse import urlparse
import requests
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import root_validator
from langchain_community.llms.vertexai import (
from langchain_community.utilities.vertexai import (
@dataclass
class _ChatHistory:
    """Represents a context and a history of messages."""
    history: List['ChatMessage'] = field(default_factory=list)
    context: Optional[str] = None