from __future__ import annotations
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import (
from uuid import UUID
from typing_extensions import TypedDict
from typing_extensions import Literal
class AnnotationQueue(BaseModel):
    """Represents an annotation queue.

    Attributes:
        id (UUID): The ID of the annotation queue.
        name (str): The name of the annotation queue.
        description (Optional[str], optional): The description of the annotation queue.
            Defaults to None.
        created_at (datetime, optional): The creation timestamp of the annotation queue.
            Defaults to the current UTC time.
        updated_at (datetime, optional): The last update timestamp of the annotation
             queue. Defaults to the current UTC time.
        tenant_id (UUID): The ID of the tenant associated with the annotation queue.
    """
    id: UUID
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tenant_id: UUID