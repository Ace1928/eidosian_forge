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
class lti(LinearTimeInvariant):
    """
    Continuous-time linear time invariant system base class.

    Parameters
    ----------
    *system : arguments
        The `lti` class can be instantiated with either 2, 3 or 4 arguments.
        The following gives the number of arguments and the corresponding
        continuous-time subclass that is created:

            * 2: `TransferFunction`:  (numerator, denominator)
            * 3: `ZerosPolesGain`: (zeros, poles, gain)
            * 4: `StateSpace`:  (A, B, C, D)

        Each argument can be an array or a sequence.

    See Also
    --------
    scipy.signal.lti
    ZerosPolesGain, StateSpace, TransferFunction, dlti

    Notes
    -----
    `lti` instances do not exist directly. Instead, `lti` creates an instance
    of one of its subclasses: `StateSpace`, `TransferFunction` or
    `ZerosPolesGain`.

    If (numerator, denominator) is passed in for ``*system``, coefficients for
    both the numerator and denominator should be specified in descending
    exponent order (e.g., ``s^2 + 3s + 5`` would be represented as ``[1, 3,
    5]``).

    Changing the value of properties that are not directly part of the current
    system representation (such as the `zeros` of a `StateSpace` system) is
    very inefficient and may lead to numerical inaccuracies. It is better to
    convert to the specific system representation first. For example, call
    ``sys = sys.to_zpk()`` before accessing/changing the zeros, poles or gain.
    """

    def __new__(cls, *system):
        """Create an instance of the appropriate subclass."""
        if cls is lti:
            N = len(system)
            if N == 2:
                return TransferFunctionContinuous.__new__(TransferFunctionContinuous, *system)
            elif N == 3:
                return ZerosPolesGainContinuous.__new__(ZerosPolesGainContinuous, *system)
            elif N == 4:
                return StateSpaceContinuous.__new__(StateSpaceContinuous, *system)
            else:
                raise ValueError('`system` needs to be an instance of `lti` or have 2, 3 or 4 arguments.')
        return super().__new__(cls)

    def __init__(self, *system):
        """
        Initialize the `lti` baseclass.

        The heavy lifting is done by the subclasses.
        """
        super().__init__(*system)

    def impulse(self, X0=None, T=None, N=None):
        """
        Return the impulse response of a continuous-time system.
        See `impulse` for details.
        """
        return impulse(self, X0=X0, T=T, N=N)

    def step(self, X0=None, T=None, N=None):
        """
        Return the step response of a continuous-time system.
        See `step` for details.
        """
        return step(self, X0=X0, T=T, N=N)

    def output(self, U, T, X0=None):
        """
        Return the response of a continuous-time system to input `U`.
        See `lsim` for details.
        """
        return lsim(self, U, T, X0=X0)

    def bode(self, w=None, n=100):
        """
        Calculate Bode magnitude and phase data of a continuous-time system.

        Returns a 3-tuple containing arrays of frequencies [rad/s], magnitude
        [dB] and phase [deg]. See `bode` for details.
        """
        return bode(self, w=w, n=n)

    def freqresp(self, w=None, n=10000):
        """
        Calculate the frequency response of a continuous-time system.

        Returns a 2-tuple containing arrays of frequencies [rad/s] and
        complex magnitude.
        See `freqresp` for details.
        """
        return freqresp(self, w=w, n=n)

    def to_discrete(self, dt, method='zoh', alpha=None):
        """Return a discretized version of the current system.

        Parameters: See `cont2discrete` for details.

        Returns
        -------
        sys: instance of `dlti`
        """
        raise NotImplementedError('to_discrete is not implemented for this system class.')