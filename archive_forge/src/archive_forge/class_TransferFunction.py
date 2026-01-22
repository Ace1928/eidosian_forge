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
class TransferFunction(LinearTimeInvariant):
    """Linear Time Invariant system class in transfer function form.

    Represents the system as the continuous-time transfer function
    :math:`H(s)=\\sum_{i=0}^N b[N-i] s^i / \\sum_{j=0}^M a[M-j] s^j` or the
    discrete-time transfer function
    :math:`H(z)=\\sum_{i=0}^N b[N-i] z^i / \\sum_{j=0}^M a[M-j] z^j`, where
    :math:`b` are elements of the numerator `num`, :math:`a` are elements of
    the denominator `den`, and ``N == len(b) - 1``, ``M == len(a) - 1``.
    `TransferFunction` systems inherit additional
    functionality from the `lti`, respectively the `dlti` classes, depending on
    which system representation is used.

    Parameters
    ----------
    *system: arguments
        The `TransferFunction` class can be instantiated with 1 or 2
        arguments. The following gives the number of input arguments and their
        interpretation:

            * 1: `lti` or `dlti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 2: array_like: (numerator, denominator)
    dt: float, optional
        Sampling time [s] of the discrete-time systems. Defaults to `None`
        (continuous-time). Must be specified as a keyword argument, for
        example, ``dt=0.1``.

    See Also
    --------
    scipy.signal.TransferFunction
    ZerosPolesGain, StateSpace, lti, dlti
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
    exponent order (e.g. ``s^2 + 3s + 5`` or ``z^2 + 3z + 5`` would be
    represented as ``[1, 3, 5]``)
    """

    def __new__(cls, *system, **kwargs):
        """Handle object conversion if input is an instance of lti."""
        if len(system) == 1 and isinstance(system[0], LinearTimeInvariant):
            return system[0].to_tf()
        if cls is TransferFunction:
            if kwargs.get('dt') is None:
                return TransferFunctionContinuous.__new__(TransferFunctionContinuous, *system, **kwargs)
            else:
                return TransferFunctionDiscrete.__new__(TransferFunctionDiscrete, *system, **kwargs)
        return super().__new__(cls)

    def __init__(self, *system, **kwargs):
        """Initialize the state space LTI system."""
        if isinstance(system[0], LinearTimeInvariant):
            return
        super().__init__(**kwargs)
        self._num = None
        self._den = None
        self.num, self.den = normalize(*system)

    def __repr__(self):
        """Return representation of the system's transfer function"""
        return '{}(\n{},\n{},\ndt: {}\n)'.format(self.__class__.__name__, repr(self.num), repr(self.den), repr(self.dt))

    @property
    def num(self):
        """Numerator of the `TransferFunction` system."""
        return self._num

    @num.setter
    def num(self, num):
        self._num = cupy.atleast_1d(num)
        if len(self.num.shape) > 1:
            self.outputs, self.inputs = self.num.shape
        else:
            self.outputs = 1
            self.inputs = 1

    @property
    def den(self):
        """Denominator of the `TransferFunction` system."""
        return self._den

    @den.setter
    def den(self, den):
        self._den = cupy.atleast_1d(den)

    def _copy(self, system):
        """
        Copy the parameters of another `TransferFunction` object

        Parameters
        ----------
        system : `TransferFunction`
            The `StateSpace` system that is to be copied

        """
        self.num = system.num
        self.den = system.den

    def to_tf(self):
        """
        Return a copy of the current `TransferFunction` system.

        Returns
        -------
        sys : instance of `TransferFunction`
            The current system (copy)

        """
        return copy.deepcopy(self)

    def to_zpk(self):
        """
        Convert system representation to `ZerosPolesGain`.

        Returns
        -------
        sys : instance of `ZerosPolesGain`
            Zeros, poles, gain representation of the current system

        """
        return ZerosPolesGain(*tf2zpk(self.num, self.den), **self._dt_dict)

    def to_ss(self):
        """
        Convert system representation to `StateSpace`.

        Returns
        -------
        sys : instance of `StateSpace`
            State space model of the current system

        """
        return StateSpace(*tf2ss(self.num, self.den), **self._dt_dict)

    @staticmethod
    def _z_to_zinv(num, den):
        """Change a transfer function from the variable `z` to `z**-1`.

        Parameters
        ----------
        num, den: 1d array_like
            Sequences representing the coefficients of the numerator and
            denominator polynomials, in order of descending degree of 'z'.
            That is, ``5z**2 + 3z + 2`` is presented as ``[5, 3, 2]``.

        Returns
        -------
        num, den: 1d array_like
            Sequences representing the coefficients of the numerator and
            denominator polynomials, in order of ascending degree of 'z**-1'.
            That is, ``5 + 3 z**-1 + 2 z**-2`` is presented as ``[5, 3, 2]``.
        """
        diff = len(num) - len(den)
        if diff > 0:
            den = cupy.hstack((cupy.zeros(diff), den))
        elif diff < 0:
            num = cupy.hstack((cupy.zeros(-diff), num))
        return (num, den)

    @staticmethod
    def _zinv_to_z(num, den):
        """Change a transfer function from the variable `z` to `z**-1`.

        Parameters
        ----------
        num, den: 1d array_like
            Sequences representing the coefficients of the numerator and
            denominator polynomials, in order of ascending degree of 'z**-1'.
            That is, ``5 + 3 z**-1 + 2 z**-2`` is presented as ``[5, 3, 2]``.

        Returns
        -------
        num, den: 1d array_like
            Sequences representing the coefficients of the numerator and
            denominator polynomials, in order of descending degree of 'z'.
            That is, ``5z**2 + 3z + 2`` is presented as ``[5, 3, 2]``.
        """
        diff = len(num) - len(den)
        if diff > 0:
            den = cupy.hstack((den, cupy.zeros(diff)))
        elif diff < 0:
            num = cupy.hstack((num, cupy.zeros(-diff)))
        return (num, den)