from sympy.combinatorics.permutations import Permutation, _af_invert, _af_rmul
from sympy.ntheory import isprime
def _strip_af(h, base, orbits, transversals, j, slp=[], slps={}):
    """
    optimized _strip, with h, transversals and result in array form
    if the stripped elements is the identity, it returns False, base_len + 1

    j    h[base[i]] == base[i] for i <= j

    """
    base_len = len(base)
    for i in range(j + 1, base_len):
        beta = h[base[i]]
        if beta == base[i]:
            continue
        if beta not in orbits[i]:
            if not slp:
                return (h, i + 1)
            return (h, i + 1, slp)
        u = transversals[i][beta]
        if h == u:
            if not slp:
                return (False, base_len + 1)
            return (False, base_len + 1, slp)
        h = _af_rmul(_af_invert(u), h)
        if slp:
            u_slp = slps[i][beta][:]
            u_slp.reverse()
            u_slp = [(i, (g,)) for g in u_slp]
            slp = u_slp + slp
    if not slp:
        return (h, base_len + 1)
    return (h, base_len + 1, slp)