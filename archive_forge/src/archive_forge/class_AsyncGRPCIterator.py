import abc
from google.api_core.page_iterator import Page
class AsyncGRPCIterator(AsyncIterator):
    """A generic class for iterating through gRPC list responses.

    .. note:: The class does not take a ``page_token`` argument because it can
        just be specified in the ``request``.

    Args:
        client (google.cloud.client.Client): The API client. This unused by
            this class, but kept to satisfy the :class:`Iterator` interface.
        method (Callable[protobuf.Message]): A bound gRPC method that should
            take a single message for the request.
        request (protobuf.Message): The request message.
        items_field (str): The field in the response message that has the
            items for the page.
        item_to_value (Callable[GRPCIterator, Any]): Callable to convert an
            item from the type in the JSON response into a native object. Will
            be called with the iterator and a single item.
        request_token_field (str): The field in the request message used to
            specify the page token.
        response_token_field (str): The field in the response message that has
            the token for the next page.
        max_results (int): The maximum number of results to fetch.

    .. autoattribute:: pages
    """
    _DEFAULT_REQUEST_TOKEN_FIELD = 'page_token'
    _DEFAULT_RESPONSE_TOKEN_FIELD = 'next_page_token'

    def __init__(self, client, method, request, items_field, item_to_value=_item_to_value_identity, request_token_field=_DEFAULT_REQUEST_TOKEN_FIELD, response_token_field=_DEFAULT_RESPONSE_TOKEN_FIELD, max_results=None):
        super().__init__(client, item_to_value, max_results=max_results)
        self._method = method
        self._request = request
        self._items_field = items_field
        self._request_token_field = request_token_field
        self._response_token_field = response_token_field

    async def _next_page(self):
        """Get the next page in the iterator.

        Returns:
            Page: The next page in the iterator or :data:`None` if
                there are no pages left.
        """
        if not self._has_next_page():
            return None
        if self.next_page_token is not None:
            setattr(self._request, self._request_token_field, self.next_page_token)
        response = await self._method(self._request)
        self.next_page_token = getattr(response, self._response_token_field)
        items = getattr(response, self._items_field)
        page = Page(self, items, self.item_to_value, raw_page=response)
        return page

    def _has_next_page(self):
        """Determines whether or not there are more pages with results.

        Returns:
            bool: Whether the iterator has more pages.
        """
        if self.page_number == 0:
            return True
        if self.max_results is not None:
            if self.num_results >= self.max_results:
                return False
        return True if self.next_page_token else False