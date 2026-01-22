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
class TransferFunctionContinuous(TransferFunction, lti):
    """
    Continuous-time Linear Time Invariant system in transfer function form.

    Represents the system as the transfer function
    :math:`H(s)=\\sum_{i=0}^N b[N-i] s^i / \\sum_{j=0}^M a[M-j] s^j`, where
    :math:`b` are elements of the numerator `num`, :math:`a` are elements of
    the denominator `den`, and ``N == len(b) - 1``, ``M == len(a) - 1``.
    Continuous-time `TransferFunction` systems inherit additional
    functionality from the `lti` class.

    Parameters
    ----------
    *system: arguments
        The `TransferFunction` class can be instantiated with 1 or 2
        arguments. The following gives the number of input arguments and their
        interpretation:

            * 1: `lti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 2: array_like: (numerator, denominator)

    See Also
    --------
    scipy.signal.TransferFunction
    ZerosPolesGain, StateSpace, lti
    tf2ss, tf2zpk, tf2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `TransferFunction` system representation (such as the `A`, `B`, `C`, `D`
    state-space matrices) is very inefficient and may lead to numerical
    inaccuracies.  It is better to convert to the specific system
    representation first. For example, call ``sys = sys.to_ss()`` before
    accessing/changing the A, B, C, D system matrices.

    If (numerator, denominator) is passed in for ``*system``, coefficients
    for both the numerator and denominator should be specified in descending
    exponent order (e.g. ``s^2 + 3s + 5`` would be represented as
    ``[1, 3, 5]``)

    """

    def to_discrete(self, dt, method='zoh', alpha=None):
        """
        Returns the discretized `TransferFunction` system.

        Parameters: See `cont2discrete` for details.

        Returns
        -------
        sys: instance of `dlti` and `StateSpace`
        """
        return TransferFunction(*cont2discrete((self.num, self.den), dt, method=method, alpha=alpha)[:-1], dt=dt)