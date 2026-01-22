import abc
def _page_iter(self, increment):
    """Generator of pages of API responses.

        Args:
            increment (bool): Flag indicating if the total number of results
                should be incremented on each page. This is useful since a page
                iterator will want to increment by results per page while an
                items iterator will want to increment per item.

        Yields:
            Page: each page of items from the API.
        """
    page = self._next_page()
    while page is not None:
        self.page_number += 1
        if increment:
            self.num_results += page.num_items
        yield page
        page = self._next_page()