import networkx as nx
from networkx.utils import py_random_state
@py_random_state(2)
def zipf_rv(alpha, xmin=1, seed=None):
    """Returns a random value chosen from the Zipf distribution.

    The return value is an integer drawn from the probability distribution

    .. math::

        p(x)=\\frac{x^{-\\alpha}}{\\zeta(\\alpha, x_{\\min})},

    where $\\zeta(\\alpha, x_{\\min})$ is the Hurwitz zeta function.

    Parameters
    ----------
    alpha : float
      Exponent value of the distribution
    xmin : int
      Minimum value
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    x : int
      Random value from Zipf distribution

    Raises
    ------
    ValueError:
      If xmin < 1 or
      If alpha <= 1

    Notes
    -----
    The rejection algorithm generates random values for a the power-law
    distribution in uniformly bounded expected time dependent on
    parameters.  See [1]_ for details on its operation.

    Examples
    --------
    >>> nx.utils.zipf_rv(alpha=2, xmin=3, seed=42)
    8

    References
    ----------
    .. [1] Luc Devroye, Non-Uniform Random Variate Generation,
       Springer-Verlag, New York, 1986.
    """
    if xmin < 1:
        raise ValueError('xmin < 1')
    if alpha <= 1:
        raise ValueError('a <= 1.0')
    a1 = alpha - 1.0
    b = 2 ** a1
    while True:
        u = 1.0 - seed.random()
        v = seed.random()
        x = int(xmin * u ** (-(1.0 / a1)))
        t = (1.0 + 1.0 / x) ** a1
        if v * x * (t - 1.0) / (b - 1.0) <= t / b:
            break
    return x