import warnings
from .._util import _any, get_backend

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
    