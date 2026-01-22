from contextlib import contextmanager
import numpy as np
from_record_like = None
def _order_like_array(ary):
    if ary.flags['F_CONTIGUOUS'] and (not ary.flags['C_CONTIGUOUS']):
        return 'F'
    else:
        return 'C'