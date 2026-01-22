import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
def _test_no_double_specialize(self, dispatcher, ty):
    with self.assertRaises(RuntimeError) as e:
        dispatcher.specialize(ty)
    self.assertIn('Dispatcher already specialized', str(e.exception))