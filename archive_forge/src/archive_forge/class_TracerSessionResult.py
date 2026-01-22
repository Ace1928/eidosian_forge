from __future__ import annotations
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import (
from uuid import UUID
from typing_extensions import TypedDict
from typing_extensions import Literal
class TracerSessionResult(TracerSession):
    """A project, hydrated with additional information.

    Sessions are also referred to as "Projects" in the UI.
    """
    run_count: Optional[int]
    'The number of runs in the project.'
    latency_p50: Optional[timedelta]
    'The median (50th percentile) latency for the project.'
    latency_p99: Optional[timedelta]
    'The 99th percentile latency for the project.'
    total_tokens: Optional[int]
    'The total number of tokens consumed in the project.'
    prompt_tokens: Optional[int]
    'The total number of prompt tokens consumed in the project.'
    completion_tokens: Optional[int]
    'The total number of completion tokens consumed in the project.'
    last_run_start_time: Optional[datetime]
    'The start time of the last run in the project.'
    feedback_stats: Optional[Dict[str, Any]]
    'Feedback stats for the project.'
    run_facets: Optional[List[Dict[str, Any]]]
    'Facets for the runs in the project.'