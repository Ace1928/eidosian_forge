from __future__ import print_function
import numpy as np
from numba import config, cuda, int32
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,

        We should only mark a kernel as cooperative and link cudadevrt if the
        kernel uses grid sync. Here we ensure that one that doesn't use grid
        synsync isn't marked as such.
        