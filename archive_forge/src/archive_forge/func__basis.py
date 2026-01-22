from sympy.polys.monomials import monomial_mul, monomial_div
def _basis(G, ring):
    """
    Computes a list of monomials which are not divisible by the leading
    monomials wrt to ``O`` of ``G``. These monomials are a basis of
    `K[X_1, \\ldots, X_n]/(G)`.
    """
    order = ring.order
    leading_monomials = [g.LM for g in G]
    candidates = [ring.zero_monom]
    basis = []
    while candidates:
        t = candidates.pop()
        basis.append(t)
        new_candidates = [_incr_k(t, k) for k in range(ring.ngens) if all((monomial_div(_incr_k(t, k), lmg) is None for lmg in leading_monomials))]
        candidates.extend(new_candidates)
        candidates.sort(key=order, reverse=True)
    basis = list(set(basis))
    return sorted(basis, key=order)