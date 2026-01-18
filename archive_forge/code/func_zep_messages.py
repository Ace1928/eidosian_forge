from __future__ import annotations
import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
@property
def zep_messages(self) -> List[Message]:
    """Retrieve summary from Zep memory"""
    zep_memory: Optional[Memory] = self._get_memory()
    if not zep_memory:
        return []
    return zep_memory.messages