from typing import Any, AsyncIterator, Awaitable, Callable, Sequence, Tuple, Optional, Iterator
from googlecloudsdk.generated_clients.gapic_clients.storage_v2.types import storage
class ListHmacKeysPager:
    """A pager for iterating through ``list_hmac_keys`` requests.

    This class thinly wraps an initial
    :class:`googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.ListHmacKeysResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``hmac_keys`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListHmacKeys`` requests and continue to iterate
    through the ``hmac_keys`` field on the
    corresponding responses.

    All the usual :class:`googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.ListHmacKeysResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(self, method: Callable[..., storage.ListHmacKeysResponse], request: storage.ListHmacKeysRequest, response: storage.ListHmacKeysResponse, *, metadata: Sequence[Tuple[str, str]]=()):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.ListHmacKeysRequest):
                The initial request object.
            response (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.ListHmacKeysResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = storage.ListHmacKeysRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[storage.ListHmacKeysResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[storage.HmacKeyMetadata]:
        for page in self.pages:
            yield from page.hmac_keys

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)