import warnings
def water_self_diffusion_coefficient(T=None, units=None, warn=True, err_mult=None):
    """
    Temperature-dependent self-diffusion coefficient of water.

    Parameters
    ----------
    T : float
        Temperature (default: in Kelvin)
    units : object (optional)
        object with attributes: Kelvin, meter, kilogram
    warn : bool (default: True)
        Emit UserWarning when outside temperature range.
    err_mult : length 2 array_like (default: None)
        Perturb parameters D0 and TS with err_mult[0]*dD0 and
        err_mult[1]*dTS respectively, where dD0 and dTS are the
        reported uncertainties in the fitted parameters. Useful
        for estimating error in diffusion coefficient.

    References
    ----------
    Temperature-dependent self-diffusion coefficients of water and six selected
        molecular liquids for calibration in accurate 1H NMR PFG measurements
        Manfred Holz, Stefan R. Heila, Antonio Saccob;
        Phys. Chem. Chem. Phys., 2000,2, 4740-4742
        http://pubs.rsc.org/en/Content/ArticleLanding/2000/CP/b005319h
        DOI: 10.1039/B005319H
    """
    if units is None:
        K = 1
        m = 1
        s = 1
    else:
        K = units.Kelvin
        m = units.meter
        s = units.second
    if T is None:
        T = 298.15 * K
    _D0 = D0 * m ** 2 * s ** (-1)
    _TS = TS * K
    if err_mult is not None:
        _dD0 = dD0 * m ** 2 * s ** (-1)
        _dTS = dTS * K
        _D0 += err_mult[0] * _dD0
        _TS += err_mult[1] * _dTS
    if warn and (_any(T < low_t_bound * K) or _any(T > high_t_bound * K)):
        warnings.warn('Temperature is outside range (0-100 degC)')
    return _D0 * (T / _TS - 1) ** gamma