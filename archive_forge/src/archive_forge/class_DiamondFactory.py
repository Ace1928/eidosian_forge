from ase.lattice.bravais import Bravais, reduceindex
import numpy as np
from ase.data import reference_states as _refstate
class DiamondFactory(FaceCenteredCubicFactory):
    """A factory for creating diamond lattices."""
    xtal_name = 'diamond'
    bravais_basis = [[0, 0, 0], [0.25, 0.25, 0.25]]