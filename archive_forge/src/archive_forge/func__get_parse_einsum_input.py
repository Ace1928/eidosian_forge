import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
@functools.lru_cache(None)
def _get_parse_einsum_input():
    try:
        from cotengra.utils import parse_einsum_input
        return parse_einsum_input
    except ImportError:
        pass
    try:
        from opt_einsum.parser import parse_einsum_input
        return parse_einsum_input
    except ImportError:
        pass
    import warnings
    warnings.warn('Could not find a full input parser for einsum expressions. Please install either cotengra or opt_einsum for advanced input formats (interleaved, ellipses, no-output).')
    return _basic_einsum_parse_input