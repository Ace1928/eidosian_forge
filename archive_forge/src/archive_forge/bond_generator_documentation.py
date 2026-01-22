import numpy as np
from ase.neighborlist import NeighborList
from ase.data import covalent_radii

    Generates bonds (lazily) one at a time, sorted by k-value (low to high).
    Here, k = d_ij / (r_i + r_j), where d_ij is the bond length and r_i and r_j
    are the covalent radii of atoms i and j.

    Parameters:

    atoms: ASE atoms object

    Returns: iterator of bonds
        A bond is a tuple with the following elements:

        k:       float   k-value
        i:       float   index of first atom
        j:       float   index of second atom
        offset:  tuple   cell offset of second atom
    