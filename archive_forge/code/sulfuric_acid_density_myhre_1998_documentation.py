import numpy as np
import warnings
from ..util import NoConvergence
Calculates the density of a solution from its concentration

    Given a function which calculates the density of a solution from the mass
    fraction of the solute, this function calculates (iteratively) the density
    of said solution for a given concentration.

    Parameters
    ----------
    conc : float (optionally with units)
        Concentration (mol / m³).
    T : float (optionally with units)
        Passed to ``rho_cb``.
    molar_mass : float (optionally with units)
        Molar mass of solute.
    rho_cb : callback
        Callback with signature f(w, T, units=None) -> rho
        (default: :func:`sulfuric_acid_density`).
    units : object (optional)
        Object with attributes: meter, kilogram, mol.
    atol : float (optionally with units)
        Convergence criterion for fixed-point iteration
        (default: 1e-3 kg/m³).
    maxiter : int
        Maximum number of iterations (when exceeded a NoConvergence exception
        is raised).
    \*\*kwargs:
        Keyword arguments passed onto ``rho_cb``.

    Returns
    -------
    Density of sulfuric acid (float of kg/m³ if T is float and units is None)

    Examples
    --------
    >>> print('%d' % density_from_concentration(400, 293))
    1021

    Raises
    ------
    chempy.util.NoConvergence:
        When maxiter is exceeded

    