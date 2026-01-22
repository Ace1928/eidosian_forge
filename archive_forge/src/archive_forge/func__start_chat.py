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
def _start_chat(self, history: _ChatHistory, **kwargs: Any) -> Union[ChatSession, CodeChatSession]:
    if not self.is_codey_model:
        return self.client.start_chat(context=history.context, message_history=history.history, **kwargs)
    else:
        return self.client.start_chat(message_history=history.history, **kwargs)