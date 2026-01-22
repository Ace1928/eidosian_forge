from ase.lattice.cubic import DiamondFactory, SimpleCubicFactory
from ase.lattice.tetragonal import SimpleTetragonalFactory
from ase.lattice.triclinic import TriclinicFactory
from ase.lattice.hexagonal import HexagonalFactory
class CsClFactory(SimpleCubicFactory):
    """A factory for creating CsCl (B2) lattices."""
    bravais_basis = [[0, 0, 0], [0.5, 0.5, 0.5]]
    element_basis = (0, 1)