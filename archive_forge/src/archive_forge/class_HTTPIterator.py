import abc
class HTTPIterator(Iterator):
    """A generic class for iterating through HTTP/JSON API list responses.

    To make an iterator work, you'll need to provide a way to convert a JSON
    item returned from the API into the object of your choice (via
    ``item_to_value``). You also may need to specify a custom ``items_key`` so
    that a given response (containing a page of results) can be parsed into an
    iterable page of the actual objects you want.

    Args:
        client (google.cloud.client.Client): The API client.
        api_request (Callable): The function to use to make API requests.
            Generally, this will be
            :meth:`google.cloud._http.JSONConnection.api_request`.
        path (str): The method path to query for the list of items.
        item_to_value (Callable[google.api_core.page_iterator.Iterator, Any]):
            Callable to convert an item from the type in the JSON response into
            a native object. Will be called with the iterator and a single
            item.
        items_key (str): The key in the API response where the list of items
            can be found.
        page_token (str): A token identifying a page in a result set to start
            fetching results from.
        page_size (int): The maximum number of results to fetch per page
        max_results (int): The maximum number of results to fetch
        extra_params (dict): Extra query string parameters for the
            API call.
        page_start (Callable[
            google.api_core.page_iterator.Iterator,
            google.api_core.page_iterator.Page, dict]): Callable to provide
            any special behavior after a new page has been created. Assumed
            signature takes the :class:`.Iterator` that started the page,
            the :class:`.Page` that was started and the dictionary containing
            the page response.
        next_token (str): The name of the field used in the response for page
            tokens.

    .. autoattribute:: pages
    """
    _DEFAULT_ITEMS_KEY = 'items'
    _PAGE_TOKEN = 'pageToken'
    _MAX_RESULTS = 'maxResults'
    _NEXT_TOKEN = 'nextPageToken'
    _RESERVED_PARAMS = frozenset([_PAGE_TOKEN])
    _HTTP_METHOD = 'GET'

    def __init__(self, client, api_request, path, item_to_value, items_key=_DEFAULT_ITEMS_KEY, page_token=None, page_size=None, max_results=None, extra_params=None, page_start=_do_nothing_page_start, next_token=_NEXT_TOKEN):
        super(HTTPIterator, self).__init__(client, item_to_value, page_token=page_token, max_results=max_results)
        self.api_request = api_request
        self.path = path
        self._items_key = items_key
        self.extra_params = extra_params
        self._page_size = page_size
        self._page_start = page_start
        self._next_token = next_token
        if self.extra_params is None:
            self.extra_params = {}
        self._verify_params()

    def _verify_params(self):
        """Verifies the parameters don't use any reserved parameter.

        Raises:
            ValueError: If a reserved parameter is used.
        """
        reserved_in_use = self._RESERVED_PARAMS.intersection(self.extra_params)
        if reserved_in_use:
            raise ValueError('Using a reserved parameter', reserved_in_use)

    def _next_page(self):
        """Get the next page in the iterator.

        Returns:
            Optional[Page]: The next page in the iterator or :data:`None` if
                there are no pages left.
        """
        if self._has_next_page():
            response = self._get_next_page_response()
            items = response.get(self._items_key, ())
            page = Page(self, items, self.item_to_value, raw_page=response)
            self._page_start(self, page, response)
            self.next_page_token = response.get(self._next_token)
            return page
        else:
            return None

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
        return self.next_page_token is not None

    def _get_query_params(self):
        """Getter for query parameters for the next request.

        Returns:
            dict: A dictionary of query parameters.
        """
        result = {}
        if self.next_page_token is not None:
            result[self._PAGE_TOKEN] = self.next_page_token
        page_size = None
        if self.max_results is not None:
            page_size = self.max_results - self.num_results
            if self._page_size is not None:
                page_size = min(page_size, self._page_size)
        elif self._page_size is not None:
            page_size = self._page_size
        if page_size is not None:
            result[self._MAX_RESULTS] = page_size
        result.update(self.extra_params)
        return result

    def _get_next_page_response(self):
        """Requests the next page from the path provided.

        Returns:
            dict: The parsed JSON response of the next page's contents.

        Raises:
            ValueError: If the HTTP method is not ``GET`` or ``POST``.
        """
        params = self._get_query_params()
        if self._HTTP_METHOD == 'GET':
            return self.api_request(method=self._HTTP_METHOD, path=self.path, query_params=params)
        elif self._HTTP_METHOD == 'POST':
            return self.api_request(method=self._HTTP_METHOD, path=self.path, data=params)
        else:
            raise ValueError('Unexpected HTTP method', self._HTTP_METHOD)