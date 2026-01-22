
    Calculates the electrical mobility through Einstein-Smoluchowski relation.

    Parameters
    ----------
    D: float with unit
        Diffusion coefficient
    charge: integer
        charge of the species
    T: float with unit
        Absolute temperature
    constants: object (optional, default: None)
        if None:
            T assumed to be in Kelvin and b0 = 1 mol/kg
        else:
            see source code for what attributes are used.
            Tip: pass quantities.constants
    units: object (optional, default: None)
        attributes accessed: meter, Kelvin and mol

    Returns
    -------
    Electrical mobility

    