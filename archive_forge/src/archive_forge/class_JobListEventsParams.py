from __future__ import annotations
from typing_extensions import TypedDict
class JobListEventsParams(TypedDict, total=False):
    after: str
    'Identifier for the last event from the previous pagination request.'
    limit: int
    'Number of events to retrieve.'