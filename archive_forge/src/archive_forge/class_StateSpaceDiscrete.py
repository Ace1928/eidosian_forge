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
class StateSpaceDiscrete(StateSpace, dlti):
    """
    Discrete-time Linear Time Invariant system in state-space form.

    Represents the system as the discrete-time difference equation
    :math:`x[k+1] = A x[k] + B u[k]`.
    `StateSpace` systems inherit additional functionality from the `dlti`
    class.

    Parameters
    ----------
    *system: arguments
        The `StateSpace` class can be instantiated with 1 or 3 arguments.
        The following gives the number of input arguments and their
        interpretation:

            * 1: `dlti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 4: array_like: (A, B, C, D)
    dt: float, optional
        Sampling time [s] of the discrete-time systems. Defaults to `True`
        (unspecified sampling time). Must be specified as a keyword argument,
        for example, ``dt=0.1``.

    See Also
    --------
    scipy.signal.StateSpaceDiscrete
    TransferFunction, ZerosPolesGain, dlti
    ss2zpk, ss2tf, zpk2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `StateSpace` system representation (such as `zeros` or `poles`) is very
    inefficient and may lead to numerical inaccuracies.  It is better to
    convert to the specific system representation first. For example, call
    ``sys = sys.to_zpk()`` before accessing/changing the zeros, poles or gain.
    """
    pass