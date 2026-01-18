from ..sage_helper import _within_sage, sage_method
from .. import SnapPy
def twisted_cochain_complex(self):
    """
        Returns chain complex of the presentation CW complex of the
        given group with coefficients twisted by self.
        """
    gens, rels, rho = (self.generators, self.relators, self)
    d1 = [[fox_derivative(R, rho, g) for g in gens] for R in rels]
    d1 = block_matrix(d1, nrows=len(rels), ncols=len(gens))
    d0 = [rho(g) - 1 for g in gens]
    d0 = block_matrix(d0, nrow=len(gens), ncols=1)
    C = ChainComplex({0: d0, 1: d1}, check=True)
    return C