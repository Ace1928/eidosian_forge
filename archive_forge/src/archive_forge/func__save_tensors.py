import contextlib
import functools
import numbers
import threading
from collections import Counter, defaultdict
from .parser import alpha_canonicalize, parse_einsum_input
def _save_tensors(*tensors):
    """Save tensors in the cache to prevent their ids from being recycled.
    This is needed to prevent false cache lookups.
    """
    cache = get_sharing_cache()
    for tensor in tensors:
        cache['tensor', id(tensor)] = tensor