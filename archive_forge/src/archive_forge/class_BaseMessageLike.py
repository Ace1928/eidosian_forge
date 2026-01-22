from __future__ import annotations
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import (
from uuid import UUID
from typing_extensions import TypedDict
from typing_extensions import Literal
@runtime_checkable
class BaseMessageLike(Protocol):
    """A protocol representing objects similar to BaseMessage."""
    content: str
    additional_kwargs: Dict

    @property
    def type(self) -> str:
        """Type of the Message, used for serialization."""