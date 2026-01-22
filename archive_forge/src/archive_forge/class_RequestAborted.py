import operator
from django.utils.hashable import make_hashable
class RequestAborted(Exception):
    """The request was closed before it was completed, or timed out."""
    pass