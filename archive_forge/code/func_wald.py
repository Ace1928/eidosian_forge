from cupy.random import _generator
from cupy import _util
def wald(mean, scale, size=None, dtype=float):
    """Wald distribution.

    Returns an array of samples drawn from the Wald distribution. Its
    probability density function is defined as

    .. math::
       f(x) = \\sqrt{\\frac{\\lambda}{2\\pi x^3}}\\
           e^{\\frac{-\\lambda(x-\\mu)^2}{2\\mu^2x}}.

    Args:
        mean (float): Parameter of the wald distribution :math:`\\mu`.
        scale (float): Parameter of the wald distribution :math:`\\lambda`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the wald distribution.

    .. seealso::
        :func:`numpy.random.wald`
    """
    rs = _generator.get_random_state()
    return rs.wald(mean, scale, size, dtype)