import warnings
from .._util import _any, get_backend
def water_permittivity(T=None, P=None, units=None, U=None, just_return_U=False, warn=True, backend=None):
    """
    Relative permittivity of water as function of temperature (K)
    and pressure (bar).

    Parameters
    ----------
    T : float
        Temperature (default: 298.15 Kelvin)
    P : float
        Pressure (default: 1 bar)
    units : object (optional)
        object with attributes: Kelvin, bar
    U : array_like (optional)
        9 parameters to the equation.
    just_return_U : bool (optional, default: False)
        Do not compute relative permittivity, just return the parameters ``U``.
    warn : bool (default: True)
        Emit UserWarning when outside temperature/pressure range.
    backend : module (default: None)
        modules which contains "exp", default: numpy, math

    Returns
    -------
    Relative permittivity of water (dielectric constant)

    References
    ----------
    Bradley, D.J.; Pitzer, K.S. `Thermodynamics of electrolytes. 12. Dielectric
        properties of water and Debye--Hueckel parameters to 350/sup
        0/C and 1 kbar`, J. Phys. Chem.; Journal Volume 83 (12)
        (1979), pp. 1599-1603,
        http://pubs.acs.org/doi/abs/10.1021/j100475a009
        DOI: 10.1021/j100475a009
    """
    be = get_backend(backend)
    if units is None:
        K = 1
        bar = 1
    else:
        K = units.kelvin
        bar = units.bar
    if T is None:
        T = 298.15 * K
    if P is None:
        P = 1 * bar
    if U is None:
        U = (342.79, -0.0050866 / K, 9.469e-07 / K ** 2, -2.0525, 3115.9 * K, -182.89 * K, -8032.5 * bar, 4214200.0 * K * bar, 2.1417 / K * bar)
    if just_return_U:
        return U
    T0 = 273.15 * K
    if warn:
        if _any(T < T0) or _any(T > T0 + 350 * K):
            warnings.warn('Outside temperature range (0-350 degC)')
        elif _any(T > T0 + 70 * K):
            if _any(P > 2000 * bar):
                warnings.warn('Outside pressure range (2000 bar)')
            elif _any(P > 5000 * bar):
                warnings.warn('Outside pressure range (5000 bar)')
    B = U[6] + U[7] / T + U[8] * T
    C = U[3] + U[4] / (U[5] + T)
    eps1000 = U[0] * be.exp(U[1] * T + U[2] * T ** 2)
    return eps1000 + C * be.log((B + P) / (B + 1000.0 * bar))