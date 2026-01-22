from __future__ import annotations
from typing import Dict, List, Union, Iterable, Optional, overload
from typing_extensions import Literal
import httpx
from ... import _legacy_response
from ..._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ..._utils import (
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from ..._streaming import Stream, AsyncStream
from ...types.chat import (
from ..._base_client import (
class CompletionsWithStreamingResponse:

    def __init__(self, completions: Completions) -> None:
        self._completions = completions
        self.create = to_streamed_response_wrapper(completions.create)