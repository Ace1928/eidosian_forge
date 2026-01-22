from __future__ import annotations
from typing import List, Iterable, Optional
from typing_extensions import Literal
import httpx
from .... import _legacy_response
from .files import (
from ...._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ...._utils import (
from ...._compat import cached_property
from ...._resource import SyncAPIResource, AsyncAPIResource
from ...._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from ....pagination import SyncCursorPage, AsyncCursorPage
from ....types.beta import (
from ...._base_client import (
class AsyncAssistantsWithStreamingResponse:

    def __init__(self, assistants: AsyncAssistants) -> None:
        self._assistants = assistants
        self.create = async_to_streamed_response_wrapper(assistants.create)
        self.retrieve = async_to_streamed_response_wrapper(assistants.retrieve)
        self.update = async_to_streamed_response_wrapper(assistants.update)
        self.list = async_to_streamed_response_wrapper(assistants.list)
        self.delete = async_to_streamed_response_wrapper(assistants.delete)

    @cached_property
    def files(self) -> AsyncFilesWithStreamingResponse:
        return AsyncFilesWithStreamingResponse(self._assistants.files)