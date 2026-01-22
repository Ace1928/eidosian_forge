from scipy import stats, integrate, special
import numpy as np
def _correction(self, x):
    """bona fide density correction

        affine shift of density to make it into a proper density

        """
    if self._corfactor != 1:
        x *= self._corfactor
    if self._corshift != 0:
        x += self._corshift
    return x