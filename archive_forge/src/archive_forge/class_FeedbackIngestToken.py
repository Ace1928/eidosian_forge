from __future__ import annotations
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import (
from uuid import UUID
from typing_extensions import TypedDict
from typing_extensions import Literal
class FeedbackIngestToken(BaseModel):
    """Represents the schema for a feedback ingest token.

    Attributes:
        id (UUID): The ID of the feedback ingest token.
        token (str): The token for ingesting feedback.
        expires_at (datetime): The expiration time of the token.
    """
    id: UUID
    url: str
    expires_at: datetime