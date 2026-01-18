import string
from ..sage_helper import _within_sage, sage_method
def test_rep(G, phialpha):

    def manually_apply_word(w):
        return prod((phialpha(x) for x in w))
    return max([univ_matrix_norm(manually_apply_word(R) - 1) for R in G.relators()])