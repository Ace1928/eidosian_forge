from math import sqrt
import numpy as np
from ase.atoms import Atoms
Create a graphene nanoribbon.

    Creates a graphene nanoribbon in the x-z plane, with the nanoribbon
    running along the z axis.

    Parameters:

    n: int
        The width of the nanoribbon.  For armchair nanoribbons, this
        n may be half-integer to repeat by half a cell.
    m: int
        The length of the nanoribbon.
    type: str
        The orientation of the ribbon.  Must be either 'zigzag'
        or 'armchair'.
    saturated: bool
        If true, hydrogen atoms are placed along the edge.
    C_H: float
        Carbon-hydrogen bond length.  Default: 1.09 Angstrom.
    C_C: float
        Carbon-carbon bond length.  Default: 1.42 Angstrom.
    vacuum: None (default) or float
        Amount of vacuum added to non-periodic directions, if present.
    magnetic: bool
        Make the edges magnetic.
    initial_mag: float
        Magnitude of magnetic moment if magnetic.
    sheet: bool
        If true, make an infinite sheet instead of a ribbon (default: False)
    