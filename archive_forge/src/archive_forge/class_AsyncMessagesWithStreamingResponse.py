from __future__ import annotations
from typing import List, Optional
from typing_extensions import Literal
import httpx
from ..... import _legacy_response
from .files import (
from ....._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ....._utils import (
from ....._compat import cached_property
from ....._resource import SyncAPIResource, AsyncAPIResource
from ....._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from .....pagination import SyncCursorPage, AsyncCursorPage
from ....._base_client import (
from .....types.beta.threads import Message, message_list_params, message_create_params, message_update_params
class AsyncMessagesWithStreamingResponse:

    def __init__(self, messages: AsyncMessages) -> None:
        self._messages = messages
        self.create = async_to_streamed_response_wrapper(messages.create)
        self.retrieve = async_to_streamed_response_wrapper(messages.retrieve)
        self.update = async_to_streamed_response_wrapper(messages.update)
        self.list = async_to_streamed_response_wrapper(messages.list)

    @cached_property
    def files(self) -> AsyncFilesWithStreamingResponse:
        return AsyncFilesWithStreamingResponse(self._messages.files)