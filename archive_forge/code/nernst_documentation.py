import math

    Calculates the Nernst potential using the Nernst equation for a particular
    ion.

    Parameters
    ----------
    ion_conc_out : float with unit
        Extracellular concentration of ion.
    ion_conc_in : float with unit
        Intracellular concentration of ion.
    charge : integer
        Charge of the ion.
    T : float with unit
        Absolute temperature.
    constants : object (optional, default: None)
        Constant attributes accessed:
            F - Faraday constant
            R - Ideal Gas constant
    units : object (optional, default: None)
        Unit attributes: coulomb, joule, kelvin, mol.
    backend : module (optional, default: math)
        Module used to calculate log using `log` method, can be substituted
        with sympy to get symbolic answers.

    Returns
    -------
    Membrane potential.

    