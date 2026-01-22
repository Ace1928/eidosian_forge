from typing import Any, AsyncIterator, Awaitable, Callable, Sequence, Tuple, Optional, Iterator
from cirq_google.cloud.quantum_v1alpha1.types import engine
from cirq_google.cloud.quantum_v1alpha1.types import quantum
class ListQuantumReservationsPager:
    """A pager for iterating through ``list_quantum_reservations`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.quantum_v1alpha1.types.ListQuantumReservationsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``reservations`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListQuantumReservations`` requests and continue to iterate
    through the ``reservations`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.quantum_v1alpha1.types.ListQuantumReservationsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(self, method: Callable[..., engine.ListQuantumReservationsResponse], request: engine.ListQuantumReservationsRequest, response: engine.ListQuantumReservationsResponse, *, metadata: Sequence[Tuple[str, str]]=()):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.quantum_v1alpha1.types.ListQuantumReservationsRequest):
                The initial request object.
            response (google.cloud.quantum_v1alpha1.types.ListQuantumReservationsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = engine.ListQuantumReservationsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[engine.ListQuantumReservationsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[quantum.QuantumReservation]:
        for page in self.pages:
            yield from page.reservations

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}<{self._response!r}>'