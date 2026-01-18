import numpy as np
from ase.cluster.cubic import FaceCenteredCubic
from ase.cluster.compounds import L1_2

    Returns Face Centered Cubic clusters of the octahedral class depending
    on the choice of cutoff.

    ============================    =======================
    Type                            Condition
    ============================    =======================
    Regular octahedron              cutoff = 0
    Truncated octahedron            cutoff > 0
    Regular truncated octahedron    length = 3 * cutoff + 1
    Cuboctahedron                   length = 2 * cutoff + 1
    ============================    =======================


    Parameters:

    symbol: string or sequence of int
        The chemical symbol or atomic number of the element(s).

    length: int
        Number of atoms on the square edges of the complete octahedron.

    cutoff (optional): int
        Number of layers cut at each vertex.

    latticeconstant (optional): float
        The lattice constant. If not given,
        then it is extracted form ase.data.

    alloy (optional): bool
        If true the L1_2 structure is used. Default is False.

    