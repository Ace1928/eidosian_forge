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
class StateSpaceContinuous(StateSpace, lti):
    """
    Continuous-time Linear Time Invariant system in state-space form.

    Represents the system as the continuous-time, first order differential
    equation :math:`\\dot{x} = A x + B u`.
    Continuous-time `StateSpace` systems inherit additional functionality
    from the `lti` class.

    Parameters
    ----------
    *system: arguments
        The `StateSpace` class can be instantiated with 1 or 3 arguments.
        The following gives the number of input arguments and their
        interpretation:

            * 1: `lti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 4: array_like: (A, B, C, D)

    See Also
    --------
    scipy.signal.StateSpaceContinuous
    TransferFunction, ZerosPolesGain, lti
    ss2zpk, ss2tf, zpk2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `StateSpace` system representation (such as `zeros` or `poles`) is very
    inefficient and may lead to numerical inaccuracies.  It is better to
    convert to the specific system representation first. For example, call
    ``sys = sys.to_zpk()`` before accessing/changing the zeros, poles or gain.
    """

    def to_discrete(self, dt, method='zoh', alpha=None):
        """
        Returns the discretized `StateSpace` system.

        Parameters: See `cont2discrete` for details.

        Returns
        -------
        sys: instance of `dlti` and `StateSpace`
        """
        return StateSpace(*cont2discrete((self.A, self.B, self.C, self.D), dt, method=method, alpha=alpha)[:-1], dt=dt)