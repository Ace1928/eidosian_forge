from ase.lattice.cubic import DiamondFactory, SimpleCubicFactory
from ase.lattice.tetragonal import SimpleTetragonalFactory
from ase.lattice.triclinic import TriclinicFactory
from ase.lattice.hexagonal import HexagonalFactory
class ZnSFactory(DiamondFactory):
    """A factory for creating ZnS (B3, Zincblende) lattices."""
    element_basis = (0, 1)