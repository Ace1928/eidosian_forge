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

        Classifies if text is potentially harmful.

        Args:
          input: The input text to classify

          model: Two content moderations models are available: `text-moderation-stable` and
              `text-moderation-latest`.

              The default is `text-moderation-latest` which will be automatically upgraded
              over time. This ensures you are always using our most accurate model. If you use
              `text-moderation-stable`, we will provide advanced notice before updating the
              model. Accuracy of `text-moderation-stable` may be slightly lower than for
              `text-moderation-latest`.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        