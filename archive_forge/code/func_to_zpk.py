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
def to_zpk(self, **kwargs):
    """
        Convert system representation to `ZerosPolesGain`.

        Parameters
        ----------
        kwargs : dict, optional
            Additional keywords passed to `ss2zpk`

        Returns
        -------
        sys : instance of `ZerosPolesGain`
            Zeros, poles, gain representation of the current system

        """
    return ZerosPolesGain(*ss2zpk(self._A, self._B, self._C, self._D, **kwargs), **self._dt_dict)