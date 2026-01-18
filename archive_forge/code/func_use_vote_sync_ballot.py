import numpy as np
from numba import cuda, int32, int64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.core import config
def use_vote_sync_ballot(ary):
    i = cuda.threadIdx.x
    ballot = cuda.ballot_sync(4294967295, True)
    ary[i] = ballot