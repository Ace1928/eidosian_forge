from ase.lattice.triclinic import TriclinicFactory
class GraphiteFactory(HexagonalFactory):
    """A factory for creating graphite lattices."""
    xtal_name = 'graphite'
    bravais_basis = [[0, 0, 0], [1.0 / 3.0, 2.0 / 3.0, 0], [1.0 / 3.0, 2.0 / 3.0, 0.5], [2.0 / 3.0, 1.0 / 3.0, 0.5]]