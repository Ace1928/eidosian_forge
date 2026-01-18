import warnings

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
    