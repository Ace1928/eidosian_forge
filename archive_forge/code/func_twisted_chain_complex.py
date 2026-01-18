from ..sage_helper import _within_sage, sage_method
from .. import SnapPy
def twisted_chain_complex(self):
    """
        Returns chain complex of the presentation CW complex of the
        given group with coefficients twisted by self.
        """
    gens, rels, rho = (self.generators, self.relators, self)
    d2 = [[fox_derivative_with_involution(R, rho, g) for R in rels] for g in gens]
    d2 = block_matrix(d2, nrows=len(gens), ncols=len(rels))
    d1 = [rho(g.swapcase()) - 1 for g in gens]
    d1 = block_matrix(d1, nrows=1, ncols=len(gens))
    C = ChainComplex({1: d1, 2: d2}, degree_of_differential=-1, check=True)
    return C