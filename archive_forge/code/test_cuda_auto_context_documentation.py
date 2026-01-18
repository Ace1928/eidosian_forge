import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
A problem was revealed by a customer that the use cuda.to_device
        does not create a CUDA context.
        This tests the problem
        