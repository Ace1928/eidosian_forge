import operator
from functools import reduce
from mmap import mmap
from numbers import Integral
import numpy as np
def _full_slicer_len(full_slicer):
    """Return length of slicer processed by ``fill_slicer``"""
    start, stop, step = (full_slicer.start, full_slicer.stop, full_slicer.step)
    if stop is None:
        stop = -1
    gap = stop - start
    if step > 0 and gap <= 0 or (step < 0 and gap >= 0):
        return 0
    return int(np.ceil(gap / step))