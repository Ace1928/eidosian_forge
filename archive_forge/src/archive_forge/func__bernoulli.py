import numpy as np
from .dtype import img_as_float
def _bernoulli(p, shape, *, rng):
    """
    Bernoulli trials at a given probability of a given size.

    This function is meant as a lower-memory alternative to calls such as
    `np.random.choice([True, False], size=image.shape, p=[p, 1-p])`.
    While `np.random.choice` can handle many classes, for the 2-class case
    (Bernoulli trials), this function is much more efficient.

    Parameters
    ----------
    p : float
        The probability that any given trial returns `True`.
    shape : int or tuple of ints
        The shape of the ndarray to return.
    rng : `numpy.random.Generator`
        ``Generator`` instance, typically obtained via `np.random.default_rng()`.

    Returns
    -------
    out : ndarray[bool]
        The results of Bernoulli trials in the given `size` where success
        occurs with probability `p`.
    """
    if p == 0:
        return np.zeros(shape, dtype=bool)
    if p == 1:
        return np.ones(shape, dtype=bool)
    return rng.random(shape) <= p