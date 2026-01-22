from ase.lattice.cubic import DiamondFactory, SimpleCubicFactory
from ase.lattice.tetragonal import SimpleTetragonalFactory
from ase.lattice.triclinic import TriclinicFactory
from ase.lattice.hexagonal import HexagonalFactory
class AuCuFactory(SimpleTetragonalFactory):
    """A factory for creating AuCu (L1_0) lattices (tetragonal symmetry)."""
    bravais_basis = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    element_basis = (0, 1, 1, 0)