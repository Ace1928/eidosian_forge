from __future__ import annotations
import json
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
from langchain_core.utils import get_from_env
Remove the session's messages from the cache.

        Raises:
            SdkException: Momento service or network error.
            Exception: Unexpected response.
        