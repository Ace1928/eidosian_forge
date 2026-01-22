from typing import Any, AsyncIterator, Awaitable, Callable, Sequence, Tuple, Optional, Iterator
from googlecloudsdk.generated_clients.gapic_clients.logging_v2.types import logging_config
class ListRecentQueriesAsyncPager:
    """A pager for iterating through ``list_recent_queries`` requests.

    This class thinly wraps an initial
    :class:`googlecloudsdk.generated_clients.gapic_clients.logging_v2.types.ListRecentQueriesResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``recent_queries`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListRecentQueries`` requests and continue to iterate
    through the ``recent_queries`` field on the
    corresponding responses.

    All the usual :class:`googlecloudsdk.generated_clients.gapic_clients.logging_v2.types.ListRecentQueriesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(self, method: Callable[..., Awaitable[logging_config.ListRecentQueriesResponse]], request: logging_config.ListRecentQueriesRequest, response: logging_config.ListRecentQueriesResponse, *, metadata: Sequence[Tuple[str, str]]=()):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (googlecloudsdk.generated_clients.gapic_clients.logging_v2.types.ListRecentQueriesRequest):
                The initial request object.
            response (googlecloudsdk.generated_clients.gapic_clients.logging_v2.types.ListRecentQueriesResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = logging_config.ListRecentQueriesRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[logging_config.ListRecentQueriesResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[logging_config.RecentQuery]:

        async def async_generator():
            async for page in self.pages:
                for response in page.recent_queries:
                    yield response
        return async_generator()

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)