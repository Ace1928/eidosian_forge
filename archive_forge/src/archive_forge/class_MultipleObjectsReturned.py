import operator
from django.utils.hashable import make_hashable
class MultipleObjectsReturned(Exception):
    """The query returned multiple objects when only one was expected."""
    pass