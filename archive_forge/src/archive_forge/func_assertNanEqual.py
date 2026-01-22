from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
def assertNanEqual(self, x, y):
    if np.isnan(x) and np.isnan(y):
        return
    self.assertEqual(x, y)