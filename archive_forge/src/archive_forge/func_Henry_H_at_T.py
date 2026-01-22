from ._util import get_backend
from .util.pyutil import defaultnamedtuple, deprecated
from .units import default_units
def Henry_H_at_T(T, H, Tderiv, T0=None, units=None, backend=None):
    """Evaluate Henry's constant H at temperature T

    Parameters
    ----------
    T: float
        Temperature (with units), assumed to be in Kelvin if ``units == None``
    H: float
        Henry's constant
    Tderiv: float (optional)
        dln(H)/d(1/T), assumed to be in Kelvin if ``units == None``.
    T0: float
        Reference temperature, assumed to be in Kelvin if ``units == None``
        default: 298.15 K
    units: object (optional)
        object with attributes: kelvin (e.g. chempy.units.default_units)
    backend : module (optional)
        module with "exp", default: numpy, math

    """
    be = get_backend(backend)
    if units is None:
        K = 1
    else:
        K = units.Kelvin
    if T0 is None:
        T0 = 298.15 * K
    return H * be.exp(Tderiv * (1 / T - 1 / T0))