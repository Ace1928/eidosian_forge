from __future__ import annotations
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import (
from uuid import UUID
from typing_extensions import TypedDict
from typing_extensions import Literal
class RunLikeDict(TypedDict, total=False):
    """Run-like dictionary, for type-hinting."""
    name: str
    run_type: RunTypeEnum
    start_time: datetime
    inputs: Optional[dict]
    outputs: Optional[dict]
    end_time: Optional[datetime]
    extra: Optional[dict]
    error: Optional[str]
    serialized: Optional[dict]
    parent_run_id: Optional[UUID]
    manifest_id: Optional[UUID]
    events: Optional[List[dict]]
    tags: Optional[List[str]]
    inputs_s3_urls: Optional[dict]
    outputs_s3_urls: Optional[dict]
    id: Optional[UUID]
    session_id: Optional[UUID]
    session_name: Optional[str]
    reference_example_id: Optional[UUID]
    input_attachments: Optional[dict]
    output_attachments: Optional[dict]
    trace_id: UUID
    dotted_order: str