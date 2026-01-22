from __future__ import annotations
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import (
from uuid import UUID
from typing_extensions import TypedDict
from typing_extensions import Literal
class LangSmithInfo(BaseModel):
    """Information about the LangSmith server."""
    version: str = ''
    'The version of the LangSmith server.'
    license_expiration_time: Optional[datetime] = None
    'The time the license will expire.'
    batch_ingest_config: Optional[BatchIngestConfig] = None