def _zpklp2lp(z, p, k, wo=1):
    """Transform a lowpass filter to a different cutoff frequency."""
    degree = _relative_degree(z, p)
    z_lp = [wo * z1 for z1 in z]
    p_lp = [wo * p1 for p1 in p]
    k_lp = k * wo ** degree
    return (z_lp, p_lp, k_lp)