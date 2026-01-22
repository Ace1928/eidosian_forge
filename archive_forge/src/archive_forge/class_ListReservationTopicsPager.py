from typing import (
from google.cloud.pubsublite_v1.types import admin
from google.cloud.pubsublite_v1.types import common
class ListReservationTopicsPager:
    """A pager for iterating through ``list_reservation_topics`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.pubsublite_v1.types.ListReservationTopicsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``topics`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListReservationTopics`` requests and continue to iterate
    through the ``topics`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.pubsublite_v1.types.ListReservationTopicsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(self, method: Callable[..., admin.ListReservationTopicsResponse], request: admin.ListReservationTopicsRequest, response: admin.ListReservationTopicsResponse, *, metadata: Sequence[Tuple[str, str]]=()):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.pubsublite_v1.types.ListReservationTopicsRequest):
                The initial request object.
            response (google.cloud.pubsublite_v1.types.ListReservationTopicsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = admin.ListReservationTopicsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[admin.ListReservationTopicsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[str]:
        for page in self.pages:
            yield from page.topics

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)