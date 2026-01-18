import numpy as np
from numba import cuda, float32, void
from numba.cuda.testing import unittest, CUDATestCase
Test issue with loop not running due to bad sign-extension at the for
        loop precondition.
        