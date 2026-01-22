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
class AsyncAssistants(AsyncAPIResource):

    @cached_property
    def files(self) -> AsyncFiles:
        return AsyncFiles(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncAssistantsWithRawResponse:
        return AsyncAssistantsWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncAssistantsWithStreamingResponse:
        return AsyncAssistantsWithStreamingResponse(self)

    async def create(self, *, model: str, description: Optional[str] | NotGiven=NOT_GIVEN, file_ids: List[str] | NotGiven=NOT_GIVEN, instructions: Optional[str] | NotGiven=NOT_GIVEN, metadata: Optional[object] | NotGiven=NOT_GIVEN, name: Optional[str] | NotGiven=NOT_GIVEN, tools: Iterable[AssistantToolParam] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Assistant:
        """
        Create an assistant with a model and instructions.

        Args:
          model: ID of the model to use. You can use the
              [List models](https://platform.openai.com/docs/api-reference/models/list) API to
              see all of your available models, or see our
              [Model overview](https://platform.openai.com/docs/models/overview) for
              descriptions of them.

          description: The description of the assistant. The maximum length is 512 characters.

          file_ids: A list of [file](https://platform.openai.com/docs/api-reference/files) IDs
              attached to this assistant. There can be a maximum of 20 files attached to the
              assistant. Files are ordered by their creation date in ascending order.

          instructions: The system instructions that the assistant uses. The maximum length is 32768
              characters.

          metadata: Set of 16 key-value pairs that can be attached to an object. This can be useful
              for storing additional information about the object in a structured format. Keys
              can be a maximum of 64 characters long and values can be a maxium of 512
              characters long.

          name: The name of the assistant. The maximum length is 256 characters.

          tools: A list of tool enabled on the assistant. There can be a maximum of 128 tools per
              assistant. Tools can be of types `code_interpreter`, `retrieval`, or `function`.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return await self._post('/assistants', body=await async_maybe_transform({'model': model, 'description': description, 'file_ids': file_ids, 'instructions': instructions, 'metadata': metadata, 'name': name, 'tools': tools}, assistant_create_params.AssistantCreateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Assistant)

    async def retrieve(self, assistant_id: str, *, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Assistant:
        """
        Retrieves an assistant.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not assistant_id:
            raise ValueError(f'Expected a non-empty value for `assistant_id` but received {assistant_id!r}')
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return await self._get(f'/assistants/{assistant_id}', options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Assistant)

    async def update(self, assistant_id: str, *, description: Optional[str] | NotGiven=NOT_GIVEN, file_ids: List[str] | NotGiven=NOT_GIVEN, instructions: Optional[str] | NotGiven=NOT_GIVEN, metadata: Optional[object] | NotGiven=NOT_GIVEN, model: str | NotGiven=NOT_GIVEN, name: Optional[str] | NotGiven=NOT_GIVEN, tools: Iterable[AssistantToolParam] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Assistant:
        """Modifies an assistant.

        Args:
          description: The description of the assistant.

        The maximum length is 512 characters.

          file_ids: A list of [File](https://platform.openai.com/docs/api-reference/files) IDs
              attached to this assistant. There can be a maximum of 20 files attached to the
              assistant. Files are ordered by their creation date in ascending order. If a
              file was previously attached to the list but does not show up in the list, it
              will be deleted from the assistant.

          instructions: The system instructions that the assistant uses. The maximum length is 32768
              characters.

          metadata: Set of 16 key-value pairs that can be attached to an object. This can be useful
              for storing additional information about the object in a structured format. Keys
              can be a maximum of 64 characters long and values can be a maxium of 512
              characters long.

          model: ID of the model to use. You can use the
              [List models](https://platform.openai.com/docs/api-reference/models/list) API to
              see all of your available models, or see our
              [Model overview](https://platform.openai.com/docs/models/overview) for
              descriptions of them.

          name: The name of the assistant. The maximum length is 256 characters.

          tools: A list of tool enabled on the assistant. There can be a maximum of 128 tools per
              assistant. Tools can be of types `code_interpreter`, `retrieval`, or `function`.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not assistant_id:
            raise ValueError(f'Expected a non-empty value for `assistant_id` but received {assistant_id!r}')
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return await self._post(f'/assistants/{assistant_id}', body=await async_maybe_transform({'description': description, 'file_ids': file_ids, 'instructions': instructions, 'metadata': metadata, 'model': model, 'name': name, 'tools': tools}, assistant_update_params.AssistantUpdateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Assistant)

    def list(self, *, after: str | NotGiven=NOT_GIVEN, before: str | NotGiven=NOT_GIVEN, limit: int | NotGiven=NOT_GIVEN, order: Literal['asc', 'desc'] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> AsyncPaginator[Assistant, AsyncCursorPage[Assistant]]:
        """Returns a list of assistants.

        Args:
          after: A cursor for use in pagination.

        `after` is an object ID that defines your place
              in the list. For instance, if you make a list request and receive 100 objects,
              ending with obj_foo, your subsequent call can include after=obj_foo in order to
              fetch the next page of the list.

          before: A cursor for use in pagination. `before` is an object ID that defines your place
              in the list. For instance, if you make a list request and receive 100 objects,
              ending with obj_foo, your subsequent call can include before=obj_foo in order to
              fetch the previous page of the list.

          limit: A limit on the number of objects to be returned. Limit can range between 1 and
              100, and the default is 20.

          order: Sort order by the `created_at` timestamp of the objects. `asc` for ascending
              order and `desc` for descending order.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return self._get_api_list('/assistants', page=AsyncCursorPage[Assistant], options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout, query=maybe_transform({'after': after, 'before': before, 'limit': limit, 'order': order}, assistant_list_params.AssistantListParams)), model=Assistant)

    async def delete(self, assistant_id: str, *, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> AssistantDeleted:
        """
        Delete an assistant.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not assistant_id:
            raise ValueError(f'Expected a non-empty value for `assistant_id` but received {assistant_id!r}')
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return await self._delete(f'/assistants/{assistant_id}', options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=AssistantDeleted)