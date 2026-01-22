from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Optional, cast
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatResult
from tenacity import (
from langchain_community.llms.utils import enforce_stop_tokens
from langchain_community.llms.yandex import _BaseYandexGPT
def _parse_chat_history(history: List[BaseMessage]) -> List[Dict[str, str]]:
    """Parse a sequence of messages into history.

    Returns:
        A list of parsed messages.
    """
    chat_history = []
    for message in history:
        content = cast(str, message.content)
        if isinstance(message, HumanMessage):
            chat_history.append(_parse_message('user', content))
        if isinstance(message, AIMessage):
            chat_history.append(_parse_message('assistant', content))
        if isinstance(message, SystemMessage):
            chat_history.append(_parse_message('system', content))
    return chat_history