import abc
from google.api_core.page_iterator import Page
def _item_to_value_identity(iterator, item):
    """An item to value transformer that returns the item un-changed."""
    return item