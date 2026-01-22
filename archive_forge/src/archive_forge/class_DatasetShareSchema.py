from __future__ import annotations
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import (
from uuid import UUID
from typing_extensions import TypedDict
from typing_extensions import Literal
class DatasetShareSchema(TypedDict, total=False):
    """Represents the schema for a dataset share.

    Attributes:
        dataset_id (UUID): The ID of the dataset.
        share_token (UUID): The token for sharing the dataset.
        url (str): The URL of the shared dataset.
    """
    dataset_id: UUID
    share_token: UUID
    url: str