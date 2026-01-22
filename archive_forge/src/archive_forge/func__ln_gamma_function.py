from math import log, exp
def _ln_gamma_function(alpha):
    """Compute the log of the gamma function for a given alpha (PRIVATE).

    Comments from Z. Yang:
    Returns ln(gamma(alpha)) for alpha>0, accurate to 10 decimal places.
    Stirling's formula is used for the central polynomial part of the procedure.
    Pike MC & Hill ID (1966) Algorithm 291: Logarithm of the gamma function.
    Communications of the Association for Computing Machinery, 9:684
    """
    if alpha <= 0:
        raise ValueError
    x = alpha
    f = 0
    if x < 7:
        f = 1
        z = x
        while z < 7:
            f *= z
            z += 1
        x = z
        f = -log(f)
    z = 1 / (x * x)
    return f + (x - 0.5) * log(x) - x + 0.918938533204673 + (((-0.000595238095238 * z + 0.000793650793651) * z - 0.002777777777778) * z + 0.083333333333333) / x