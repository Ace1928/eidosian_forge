import warnings
import copy
from math import sqrt
import cupy
from cupyx.scipy import linalg
from cupyx.scipy.interpolate import make_interp_spline
from cupyx.scipy.linalg import expm, block_diag
from cupyx.scipy.signal._lti_conversion import (
from cupyx.scipy.signal._iir_filter_conversions import (
from cupyx.scipy.signal._filter_design import (
def _as_ss(self):
    """Convert to `StateSpace` system, without copying.

        Returns
        -------
        sys: StateSpace
            The `StateSpace` system. If the class is already an instance of
            `StateSpace` then this instance is returned.
        """
    if isinstance(self, StateSpace):
        return self
    else:
        return self.to_ss()