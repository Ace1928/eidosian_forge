from __future__ import annotations
import httpx
from .. import _legacy_response
from ..types import Model, ModelDeleted
from .._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from .._compat import cached_property
from .._resource import SyncAPIResource, AsyncAPIResource
from .._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from ..pagination import SyncPage, AsyncPage
from .._base_client import (
class ModelsWithStreamingResponse:

    def __init__(self, models: Models) -> None:
        self._models = models
        self.retrieve = to_streamed_response_wrapper(models.retrieve)
        self.list = to_streamed_response_wrapper(models.list)
        self.delete = to_streamed_response_wrapper(models.delete)