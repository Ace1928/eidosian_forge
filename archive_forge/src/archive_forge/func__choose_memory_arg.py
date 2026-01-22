from collections import namedtuple
from decimal import Decimal
import numpy as np
from . import backends, blas, helpers, parser, paths, sharing
def _choose_memory_arg(memory_limit, size_list):
    if memory_limit == 'max_input':
        return max(size_list)
    if memory_limit is None:
        return None
    if memory_limit < 1:
        if memory_limit == -1:
            return None
        else:
            raise ValueError('Memory limit must be larger than 0, or -1')
    return int(memory_limit)