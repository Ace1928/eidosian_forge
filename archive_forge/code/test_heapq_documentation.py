import heapq as hq
import itertools
import numpy as np
from numba import jit, typed
from numba.tests.support import TestCase, MemoryLeakMixin
Test heapq with typed lists