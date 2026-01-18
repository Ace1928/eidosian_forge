import multiprocessing as mp
import itertools
import traceback
import pickle
import numpy as np
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import (skip_on_arm, skip_on_cudasim,
from numba.tests.support import linux_only, windows_only
import unittest
def variants(self):
    indices = (None, slice(3, None), slice(3, 8), slice(None, 8))
    foreigns = (False, True)
    return itertools.product(indices, foreigns)