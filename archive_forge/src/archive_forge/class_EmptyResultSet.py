import operator
from django.utils.hashable import make_hashable
class EmptyResultSet(Exception):
    """A database query predicate is impossible."""
    pass