import numpy as np
from ase.eos import birchmurnaghan
Calculate Delta-value between two equation of states.

    .. seealso:: https://github.com/molmod/DeltaCodesDFT

    Parameters
    ----------
    v1,v2: float
        Volume per atom.
    B1,B2: float
        Bulk-modulus (in eV/Ang^3).
    Bp1,Bp2: float
        Pressure derivative of bulk-modulus.
    symmetric: bool
        Default is to calculate a symmetric delta.

    Returns
    -------
    delta: float
        Delta value in eV/atom.
    