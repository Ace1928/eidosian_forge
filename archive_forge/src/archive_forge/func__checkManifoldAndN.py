from . import matrix
from .polynomial import Polynomial
from ..pari import pari
def _checkManifoldAndN(self, manifold, N):
    if self._manifold is not None:
        assert manifold == self._manifold, 'PtolemyGeneralizedObstructionClass for wrong manifold'
    if self._N is not None:
        assert N == self._N, 'PtolemyGeneralizedObstructionClass for wrong N'
    assert len(self.H2_class) == 2 * manifold.num_tetrahedra(), 'PtolemyGeneralizedObstructionClass does not match number of face classes'
    chain_d3, dummy_rows, dummy_columns = manifold._ptolemy_equations_boundary_map_3()
    cochain_d2 = matrix.matrix_transpose(chain_d3)
    assert matrix.is_vector_zero(matrix.vector_modulo(matrix.matrix_mult_vector(cochain_d2, self.H2_class), N)), 'PtolemyGeneralizedObstructionClass not in kernel of d2'