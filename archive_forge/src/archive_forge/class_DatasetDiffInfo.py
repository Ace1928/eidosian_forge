from __future__ import annotations
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import (
from uuid import UUID
from typing_extensions import TypedDict
from typing_extensions import Literal
class DatasetDiffInfo(BaseModel):
    """Represents the difference information between two datasets.

    Attributes:
        examples_modified (List[UUID]): A list of UUIDs representing
            the modified examples.
        examples_added (List[UUID]): A list of UUIDs representing
            the added examples.
        examples_removed (List[UUID]): A list of UUIDs representing
            the removed examples.
    """
    examples_modified: List[UUID]
    examples_added: List[UUID]
    examples_removed: List[UUID]