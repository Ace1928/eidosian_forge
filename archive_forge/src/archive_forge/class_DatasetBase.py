from __future__ import annotations
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import (
from uuid import UUID
from typing_extensions import TypedDict
from typing_extensions import Literal
class DatasetBase(BaseModel):
    """Dataset base model."""
    name: str
    description: Optional[str] = None
    data_type: Optional[DataType] = None

    class Config:
        """Configuration class for the schema."""
        frozen = True