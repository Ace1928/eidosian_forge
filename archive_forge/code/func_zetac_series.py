def zetac_series(N):
    coeffs = []
    with mpmath.workdps(100):
        coeffs.append(-1.5)
        for n in range(1, N):
            coeff = mpmath.diff(mpmath.zeta, 0, n) / mpmath.factorial(n)
            coeffs.append(coeff)
    return coeffs