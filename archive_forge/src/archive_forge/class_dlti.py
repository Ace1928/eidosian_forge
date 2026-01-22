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
class dlti(LinearTimeInvariant):
    """
    Discrete-time linear time invariant system base class.

    Parameters
    ----------
    *system: arguments
        The `dlti` class can be instantiated with either 2, 3 or 4 arguments.
        The following gives the number of arguments and the corresponding
        discrete-time subclass that is created:

            * 2: `TransferFunction`:  (numerator, denominator)
            * 3: `ZerosPolesGain`: (zeros, poles, gain)
            * 4: `StateSpace`:  (A, B, C, D)

        Each argument can be an array or a sequence.
    dt: float, optional
        Sampling time [s] of the discrete-time systems. Defaults to ``True``
        (unspecified sampling time). Must be specified as a keyword argument,
        for example, ``dt=0.1``.

    See Also
    --------
    scipy.signal.dlti
    ZerosPolesGain, StateSpace, TransferFunction, lti

    Notes
    -----
    `dlti` instances do not exist directly. Instead, `dlti` creates an instance
    of one of its subclasses: `StateSpace`, `TransferFunction` or
    `ZerosPolesGain`.

    Changing the value of properties that are not directly part of the current
    system representation (such as the `zeros` of a `StateSpace` system) is
    very inefficient and may lead to numerical inaccuracies.  It is better to
    convert to the specific system representation first. For example, call
    ``sys = sys.to_zpk()`` before accessing/changing the zeros, poles or gain.

    If (numerator, denominator) is passed in for ``*system``, coefficients for
    both the numerator and denominator should be specified in descending
    exponent order (e.g., ``z^2 + 3z + 5`` would be represented as ``[1, 3,
    5]``).
    """

    def __new__(cls, *system, **kwargs):
        """Create an instance of the appropriate subclass."""
        if cls is dlti:
            N = len(system)
            if N == 2:
                return TransferFunctionDiscrete.__new__(TransferFunctionDiscrete, *system, **kwargs)
            elif N == 3:
                return ZerosPolesGainDiscrete.__new__(ZerosPolesGainDiscrete, *system, **kwargs)
            elif N == 4:
                return StateSpaceDiscrete.__new__(StateSpaceDiscrete, *system, **kwargs)
            else:
                raise ValueError('`system` needs to be an instance of `dlti` or have 2, 3 or 4 arguments.')
        return super().__new__(cls)

    def __init__(self, *system, **kwargs):
        """
        Initialize the `lti` baseclass.

        The heavy lifting is done by the subclasses.
        """
        dt = kwargs.pop('dt', True)
        super().__init__(*system, **kwargs)
        self.dt = dt

    @property
    def dt(self):
        """Return the sampling time of the system."""
        return self._dt

    @dt.setter
    def dt(self, dt):
        self._dt = dt

    def impulse(self, x0=None, t=None, n=None):
        """
        Return the impulse response of the discrete-time `dlti` system.
        See `dimpulse` for details.
        """
        return dimpulse(self, x0=x0, t=t, n=n)

    def step(self, x0=None, t=None, n=None):
        """
        Return the step response of the discrete-time `dlti` system.
        See `dstep` for details.
        """
        return dstep(self, x0=x0, t=t, n=n)

    def output(self, u, t, x0=None):
        """
        Return the response of the discrete-time system to input `u`.
        See `dlsim` for details.
        """
        return dlsim(self, u, t, x0=x0)

    def bode(self, w=None, n=100):
        """
        Calculate Bode magnitude and phase data of a discrete-time system.

        Returns a 3-tuple containing arrays of frequencies [rad/s], magnitude
        [dB] and phase [deg]. See `dbode` for details.
        """
        return dbode(self, w=w, n=n)

    def freqresp(self, w=None, n=10000, whole=False):
        """
        Calculate the frequency response of a discrete-time system.

        Returns a 2-tuple containing arrays of frequencies [rad/s] and
        complex magnitude.
        See `dfreqresp` for details.

        """
        return dfreqresp(self, w=w, n=n, whole=whole)