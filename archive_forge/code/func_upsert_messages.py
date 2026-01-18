from __future__ import annotations
import logging
from types import TracebackType
from typing import TYPE_CHECKING, Any, List, Optional, Type
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
def upsert_messages(self) -> None:
    """Update the cosmosdb item."""
    if not self._container:
        raise ValueError('Container not initialized')
    self._container.upsert_item(body={'id': self.session_id, 'user_id': self.user_id, 'messages': messages_to_dict(self.messages)})