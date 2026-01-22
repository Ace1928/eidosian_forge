from __future__ import annotations
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import (
from uuid import UUID
from typing_extensions import TypedDict
from typing_extensions import Literal
class RunEvent(TypedDict, total=False):
    """Run event schema."""
    name: str
    'Type of event.'
    time: Union[datetime, str]
    'Time of the event.'
    kwargs: Optional[Dict[str, Any]]
    'Additional metadata for the event.'