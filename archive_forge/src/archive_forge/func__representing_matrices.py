from sympy.polys.monomials import monomial_mul, monomial_div
def _representing_matrices(basis, G, ring):
    """
    Compute the matrices corresponding to the linear maps `m \\mapsto
    x_i m` for all variables `x_i`.
    """
    domain = ring.domain
    u = ring.ngens - 1

    def var(i):
        return tuple([0] * i + [1] + [0] * (u - i))

    def representing_matrix(m):
        M = [[domain.zero] * len(basis) for _ in range(len(basis))]
        for i, v in enumerate(basis):
            r = ring.term_new(monomial_mul(m, v), domain.one).rem(G)
            for monom, coeff in r.terms():
                j = basis.index(monom)
                M[j][i] = coeff
        return M
    return [representing_matrix(var(i)) for i in range(u + 1)]