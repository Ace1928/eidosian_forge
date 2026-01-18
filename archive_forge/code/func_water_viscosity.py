import warnings
from .._util import _any
def water_viscosity(T=None, eta20=None, units=None, warn=True):
    """Viscosity of water (cP) as function of temperature (K)

    Parameters
    ----------
    T : float
        Temperature (in Kelvin) (default: 298.15 K)
    eta20 : float
        Viscosity of water at 20 degree Celsius.
    units : object (optional)
        object with attributes: kelvin & centipoise
    warn : bool
        Emit UserWarning when outside temperature range.

    Returns
    -------
    Water viscosity at temperature ``T``.

    """
    if units is None:
        cP = 1
        K = 1
    else:
        cP = units.centipoise
        K = units.kelvin
    if T is None:
        T = 298.15 * K
    if eta20 is None:
        eta20 = eta20_cP * cP
    t = T - 273.15 * K
    if warn and (_any(t < 0 * K) or _any(t > 100 * K)):
        warnings.warn('Temperature is outside range (0-100 degC)')
    return eta20 * 10 ** ((A * (20 - t) - B * (t - 20) ** 2) / (t + C))