from .._util import get_backend
from ..util.regression import least_squares
from ..util.pyutil import defaultnamedtuple
from ..units import default_constants, default_units, format_string, patched_numpy
class ArrheniusParam(defaultnamedtuple('ArrheniusParam', 'A Ea ref', [None])):
    """Kinetic data in the form of an Arrhenius parameterisation

    Parameters
    ----------
    Ea: float
        activation energy
    A: float
        preexponential prefactor (Arrhenius type eq.)
    ref: object (default: None)
        arbitrary reference (e.g. string representing citation key)

    Examples
    --------
    >>> k = ArrheniusParam(1e13, 40e3)
    >>> '%.5g' % k(298.15)
    '9.8245e+05'

    """

    def html(self, fmt):
        return '%s exp((%s)/(RT))' % (fmt(self.A), fmt(self.Ea))

    def unicode(self, fmt):
        return '%s exp((%s)/(RT))' % (fmt(self.A), fmt(self.Ea))

    @classmethod
    def from_rateconst_at_T(cls, Ea, T_k, backend=None, constants=None, units=None, **kwargs):
        """Constructs an instance from a known rate constant at a given temperature.

        Parameters
        ----------
        Ea : float
            Activation energy.
        T_k : tuple of two floats
            Temperature & rate constant.

        """
        T, k = T_k
        R = _get_R(constants, units)
        if backend is None:
            from chempy.units import patched_numpy as backend
        return cls(k * backend.exp(Ea / R / T), Ea, **kwargs)

    @classmethod
    def from_fit_of_data(cls, T, k, kerr=None, **kwargs):
        args, vcv = fit_arrhenius_equation(T, k, kerr)
        return cls(*args, **kwargs)

    def __call__(self, T, constants=None, units=None, backend=None):
        """Evaluates the arrhenius equation for a specified state

        Parameters
        ----------
        T: float
        constants: module (optional)
        units: module (optional)
        backend: module (default: math)

        See also
        --------
        chempy.arrhenius.arrhenius_equation : the function called here.

        """
        return arrhenius_equation(self.A, self.Ea, T, constants=constants, units=units, backend=backend)

    def Ea_over_R(self, constants, units, backend=None):
        return self.Ea / _get_R(constants, units)

    def as_RateExpr(self, unique_keys=None, constants=None, units=None, backend=None):
        from .rates import Arrhenius, MassAction
        args = [self.A, self.Ea_over_R(constants, units)]
        return MassAction(Arrhenius(args, unique_keys))

    def format(self, precision, tex=False):
        try:
            str_A, str_A_unit = format_string(self.A, precision, tex)
            str_Ea, str_Ea_unit = format_string(self.Ea, precision, tex)
        except Exception:
            str_A, str_A_unit = (precision.format(self.A), '-')
            str_Ea, str_Ea_unit = (precision.format(self.Ea), '-')
        return ((str_A, str_A_unit), (str_Ea, str_Ea_unit))

    def equation_as_string(self, precision, tex=False):
        (str_A, str_A_unit), (str_Ea, str_Ea_unit) = self.format(precision, tex)
        if tex:
            return ('{}\\exp \\left(-\\frac{{{}}}{{RT}} \\right)'.format(str_A, str_Ea + ' ' + str_Ea_unit), str_A_unit)
        else:
            return ('{}*exp(-{}/(R*T))'.format(str_A, str_Ea + ' ' + str_Ea_unit), str_A_unit)

    def __str__(self):
        return ' '.join(self.equation_as_string('%.5g'))