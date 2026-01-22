import re
import warnings
from numpy import asarray, array, zeros, isscalar, real, imag, vstack
from . import _vode
from . import _dop
from . import _lsoda
def _determine_mf_and_set_bands(self, has_jac):
    """
        Determine the `MF` parameter (Method Flag) for the Fortran subroutine `dvode`.

        In the Fortran code, the legal values of `MF` are:
            10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25,
            -11, -12, -14, -15, -21, -22, -24, -25
        but this Python wrapper does not use negative values.

        Returns

            mf  = 10*self.meth + miter

        self.meth is the linear multistep method:
            self.meth == 1:  method="adams"
            self.meth == 2:  method="bdf"

        miter is the correction iteration method:
            miter == 0:  Functional iteration; no Jacobian involved.
            miter == 1:  Chord iteration with user-supplied full Jacobian.
            miter == 2:  Chord iteration with internally computed full Jacobian.
            miter == 3:  Chord iteration with internally computed diagonal Jacobian.
            miter == 4:  Chord iteration with user-supplied banded Jacobian.
            miter == 5:  Chord iteration with internally computed banded Jacobian.

        Side effects: If either self.mu or self.ml is not None and the other is None,
        then the one that is None is set to 0.
        """
    jac_is_banded = self.mu is not None or self.ml is not None
    if jac_is_banded:
        if self.mu is None:
            self.mu = 0
        if self.ml is None:
            self.ml = 0
    if has_jac:
        if jac_is_banded:
            miter = 4
        else:
            miter = 1
    elif jac_is_banded:
        if self.ml == self.mu == 0:
            miter = 3
        else:
            miter = 5
    elif self.with_jacobian:
        miter = 2
    else:
        miter = 0
    mf = 10 * self.meth + miter
    return mf