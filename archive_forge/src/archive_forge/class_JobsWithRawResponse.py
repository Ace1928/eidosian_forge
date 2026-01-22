from __future__ import annotations
from typing import Union, Optional
from typing_extensions import Literal
import httpx
from ... import _legacy_response
from ..._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ..._utils import (
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from ...pagination import SyncCursorPage, AsyncCursorPage
from ..._base_client import (
from ...types.fine_tuning import (
class JobsWithRawResponse:

    def __init__(self, jobs: Jobs) -> None:
        self._jobs = jobs
        self.create = _legacy_response.to_raw_response_wrapper(jobs.create)
        self.retrieve = _legacy_response.to_raw_response_wrapper(jobs.retrieve)
        self.list = _legacy_response.to_raw_response_wrapper(jobs.list)
        self.cancel = _legacy_response.to_raw_response_wrapper(jobs.cancel)
        self.list_events = _legacy_response.to_raw_response_wrapper(jobs.list_events)