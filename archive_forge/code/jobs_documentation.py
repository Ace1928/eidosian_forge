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

        Get status updates for a fine-tuning job.

        Args:
          after: Identifier for the last event from the previous pagination request.

          limit: Number of events to retrieve.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        