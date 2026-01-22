import math
from .._util import get_backend
from ..util.pyutil import defaultnamedtuple
from ..units import default_units, Backend, default_constants, format_string
from .arrhenius import _get_R, _fit
class EyringParam(defaultnamedtuple('EyringParam', 'dH dS ref', [None])):
    """Kinetic data in the form of an Eyring parameterisation

    Parameters
    ----------
    dH : float
        Enthalpy of activation.
    dS : float
        Entropy of activation.
    ref: object (default: None)
        arbitrary reference (e.g. citation key or dict with bibtex entries)

    Examples
    --------
    >>> k = EyringParam(72e3, 61.4)
    >>> '%.5g' % k(298.15)
    '2435.4'

    """

    def __call__(self, T, constants=None, units=None, backend=None):
        """Evaluates the Eyring equation for a specified state

        Parameters
        ----------
        T : float
        constants : module (optional)
        units : module (optional)
        backend : module (default: math)

        See also
        --------
        chempy.eyring.eyring_equation : the function used here

        """
        return eyring_equation(self.dH, self.dS, T, constants=constants, units=units, backend=backend)

    def kB_h_times_exp_dS_R(self, constants=None, units=None, backend=math):
        R = _get_R(constants, units)
        kB_over_h = _get_kB_over_h(constants, units)
        return kB_over_h * backend.exp(self.dS / R)

    def dH_over_R(self, constants=None, units=None, backend=None):
        R = _get_R(constants, units)
        return self.dH / R

    def as_RateExpr(self, unique_keys=None, constants=None, units=None, backend=math):
        from .rates import Eyring, MassAction
        args = [self.kB_h_times_exp_dS_R(constants, units, backend), self.dH_over_R(constants, units)]
        return MassAction(Eyring(args, unique_keys))

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
            return ('\\frac{{k_B T}}{{h}}\\exp \\left(\\frac{{{}}}{{R}} \\right) \\exp \\left(-\\frac{{{}}}{{RT}} \\right)'.format(str_A, str_Ea + ' ' + str_Ea_unit), str_A_unit)
        else:
            return ('kB*T/h*exp({}/R)*exp(-{}/(R*T))'.format(str_A, str_Ea + ' ' + str_Ea_unit), str_A_unit)

    def __str__(self):
        return self.equation_as_string('%.5g')