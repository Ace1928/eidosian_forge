import numpy as np
from ase.cluster import Icosahedron
from pytest import mark

    Test that a static relaxation that requires multiple neighbor list
    rebuilds can be carried out successfully.  This is verified by relaxing
    an icosahedral cluster of atoms and checking that the relaxed energy
    matches a known precomputed value for an example model.
    