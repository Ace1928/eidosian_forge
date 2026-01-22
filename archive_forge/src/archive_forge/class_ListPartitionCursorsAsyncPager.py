from typing import (
from google.cloud.pubsublite_v1.types import cursor
class ListPartitionCursorsAsyncPager:
    """A pager for iterating through ``list_partition_cursors`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.pubsublite_v1.types.ListPartitionCursorsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``partition_cursors`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListPartitionCursors`` requests and continue to iterate
    through the ``partition_cursors`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.pubsublite_v1.types.ListPartitionCursorsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(self, method: Callable[..., Awaitable[cursor.ListPartitionCursorsResponse]], request: cursor.ListPartitionCursorsRequest, response: cursor.ListPartitionCursorsResponse, *, metadata: Sequence[Tuple[str, str]]=()):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.pubsublite_v1.types.ListPartitionCursorsRequest):
                The initial request object.
            response (google.cloud.pubsublite_v1.types.ListPartitionCursorsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = cursor.ListPartitionCursorsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[cursor.ListPartitionCursorsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[cursor.PartitionCursor]:

        async def async_generator():
            async for page in self.pages:
                for response in page.partition_cursors:
                    yield response
        return async_generator()

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)