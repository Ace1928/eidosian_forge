def zpkfreqz(z, p, k, worN=None):
    """
    Frequency response of a filter in zpk format, using mpmath.

    This is the same calculation as scipy.signal.freqz, but the input is in
    zpk format, the calculation is performed using mpath, and the results are
    returned in lists instead of NumPy arrays.
    """
    if worN is None or isinstance(worN, int):
        N = worN or 512
        ws = [mpmath.pi * mpmath.mpf(j) / N for j in range(N)]
    else:
        ws = worN
    h = []
    for wk in ws:
        zm1 = mpmath.exp(1j * wk)
        numer = _prod([zm1 - t for t in z])
        denom = _prod([zm1 - t for t in p])
        hk = k * numer / denom
        h.append(hk)
    return (ws, h)