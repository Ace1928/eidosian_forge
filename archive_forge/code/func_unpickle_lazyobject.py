import copy
import itertools
import operator
from functools import wraps
def unpickle_lazyobject(wrapped):
    """
    Used to unpickle lazy objects. Just return its argument, which will be the
    wrapped object.
    """
    return wrapped