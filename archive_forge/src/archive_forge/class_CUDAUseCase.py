from numba import cuda
from numba.cuda.testing import CUDATestCase
import numpy as np
import sys
class CUDAUseCase(UseCase):

    def _call(self, ret, *args):
        self._func[1, 1](ret, *args)