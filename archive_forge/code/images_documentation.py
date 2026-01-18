from __future__ import annotations
from typing import Union, Mapping, Optional, cast
from typing_extensions import Literal
import httpx
from .. import _legacy_response
from ..types import (
from .._types import NOT_GIVEN, Body, Query, Headers, NotGiven, FileTypes
from .._utils import (
from .._compat import cached_property
from .._resource import SyncAPIResource, AsyncAPIResource
from .._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from .._base_client import (

        Creates an image given a prompt.

        Args:
          prompt: A text description of the desired image(s). The maximum length is 1000
              characters for `dall-e-2` and 4000 characters for `dall-e-3`.

          model: The model to use for image generation.

          n: The number of images to generate. Must be between 1 and 10. For `dall-e-3`, only
              `n=1` is supported.

          quality: The quality of the image that will be generated. `hd` creates images with finer
              details and greater consistency across the image. This param is only supported
              for `dall-e-3`.

          response_format: The format in which the generated images are returned. Must be one of `url` or
              `b64_json`. URLs are only valid for 60 minutes after the image has been
              generated.

          size: The size of the generated images. Must be one of `256x256`, `512x512`, or
              `1024x1024` for `dall-e-2`. Must be one of `1024x1024`, `1792x1024`, or
              `1024x1792` for `dall-e-3` models.

          style: The style of the generated images. Must be one of `vivid` or `natural`. Vivid
              causes the model to lean towards generating hyper-real and dramatic images.
              Natural causes the model to produce more natural, less hyper-real looking
              images. This param is only supported for `dall-e-3`.

          user: A unique identifier representing your end-user, which can help OpenAI to monitor
              and detect abuse.
              [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        