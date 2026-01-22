from warnings import warn
import numpy as np
from scipy._lib._util import VisibleDeprecationWarning
from ._sputils import (asmatrix, check_reshape_kwargs, check_shape,
from ._matrix import spmatrix
def _rmul_dispatch(self, other):
    if isscalarlike(other):
        return self._mul_scalar(other)
    else:
        try:
            tr = other.transpose()
        except AttributeError:
            tr = np.asarray(other).transpose()
        ret = self.transpose()._mul_dispatch(tr)
        if ret is NotImplemented:
            return NotImplemented
        return ret.transpose()