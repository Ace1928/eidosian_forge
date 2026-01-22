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
class Models(SyncAPIResource):

    @cached_property
    def with_raw_response(self) -> ModelsWithRawResponse:
        return ModelsWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> ModelsWithStreamingResponse:
        return ModelsWithStreamingResponse(self)

    def retrieve(self, model: str, *, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Model:
        """
        Retrieves a model instance, providing basic information about the model such as
        the owner and permissioning.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not model:
            raise ValueError(f'Expected a non-empty value for `model` but received {model!r}')
        return self._get(f'/models/{model}', options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Model)

    def list(self, *, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> SyncPage[Model]:
        """
        Lists the currently available models, and provides basic information about each
        one such as the owner and availability.
        """
        return self._get_api_list('/models', page=SyncPage[Model], options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), model=Model)

    def delete(self, model: str, *, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> ModelDeleted:
        """Delete a fine-tuned model.

        You must have the Owner role in your organization to
        delete a model.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not model:
            raise ValueError(f'Expected a non-empty value for `model` but received {model!r}')
        return self._delete(f'/models/{model}', options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=ModelDeleted)