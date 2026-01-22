from __future__ import annotations
from typing import List, Union
from typing_extensions import Literal
import httpx
from .. import _legacy_response
from ..types import ModerationCreateResponse, moderation_create_params
from .._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from .._utils import (
from .._compat import cached_property
from .._resource import SyncAPIResource, AsyncAPIResource
from .._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from .._base_client import (
class ModerationsWithStreamingResponse:

    def __init__(self, moderations: Moderations) -> None:
        self._moderations = moderations
        self.create = to_streamed_response_wrapper(moderations.create)