import abc
from google.api_core.page_iterator import Page
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