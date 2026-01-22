from __future__ import annotations
from typing import Iterable, Optional, overload
from functools import partial
from typing_extensions import Literal
import httpx
from .... import _legacy_response
from .runs import (
from .messages import (
from ...._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ...._utils import (
from .runs.runs import Runs, AsyncRuns
from ...._compat import cached_property
from ...._resource import SyncAPIResource, AsyncAPIResource
from ...._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from ...._streaming import Stream, AsyncStream
from ....types.beta import (
from ...._base_client import (
from ....lib.streaming import (
from .messages.messages import Messages, AsyncMessages
from ....types.beta.threads import Run
class ThreadsWithStreamingResponse:

    def __init__(self, threads: Threads) -> None:
        self._threads = threads
        self.create = to_streamed_response_wrapper(threads.create)
        self.retrieve = to_streamed_response_wrapper(threads.retrieve)
        self.update = to_streamed_response_wrapper(threads.update)
        self.delete = to_streamed_response_wrapper(threads.delete)
        self.create_and_run = to_streamed_response_wrapper(threads.create_and_run)

    @cached_property
    def runs(self) -> RunsWithStreamingResponse:
        return RunsWithStreamingResponse(self._threads.runs)

    @cached_property
    def messages(self) -> MessagesWithStreamingResponse:
        return MessagesWithStreamingResponse(self._threads.messages)