import numpy as np
from ase import Atoms
from pytest import mark

    To test that the correct energy/forces/stress can be computed using a
    model that implements multiple cutoffs.  This is done by construct a 10
    Angstrom x 10 Angstrom x 10 Angstrom non-periodic cell filled with 15
    randomly positioned atoms and requesting tha tthe model compute the
    energy, forces, and virial stress.  The energy returned by the model is
    compared to a known precomputed value, while the forces and stress
    returned are compared to numerical estimates via finite difference.
    