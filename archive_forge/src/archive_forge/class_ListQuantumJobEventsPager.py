from typing import Any, AsyncIterator, Awaitable, Callable, Sequence, Tuple, Optional, Iterator
from cirq_google.cloud.quantum_v1alpha1.types import engine
from cirq_google.cloud.quantum_v1alpha1.types import quantum
class ListQuantumJobEventsPager:
    """A pager for iterating through ``list_quantum_job_events`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.quantum_v1alpha1.types.ListQuantumJobEventsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``events`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListQuantumJobEvents`` requests and continue to iterate
    through the ``events`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.quantum_v1alpha1.types.ListQuantumJobEventsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(self, method: Callable[..., engine.ListQuantumJobEventsResponse], request: engine.ListQuantumJobEventsRequest, response: engine.ListQuantumJobEventsResponse, *, metadata: Sequence[Tuple[str, str]]=()):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.quantum_v1alpha1.types.ListQuantumJobEventsRequest):
                The initial request object.
            response (google.cloud.quantum_v1alpha1.types.ListQuantumJobEventsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = engine.ListQuantumJobEventsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[engine.ListQuantumJobEventsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[quantum.QuantumJobEvent]:
        for page in self.pages:
            yield from page.events

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}<{self._response!r}>'