from sympy.polys.monomials import monomial_mul, monomial_div
def _matrix_mul(M, v):
    return [sum([row[i] * v[i] for i in range(len(v))]) for row in M]