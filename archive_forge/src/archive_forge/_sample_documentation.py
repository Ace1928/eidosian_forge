from cupy import _core
from cupy._creation import basic
from cupy.random import _distributions
from cupy.random import _generator
Returns an array from multinomial distribution.

    Args:
        n (int): Number of trials.
        pvals (cupy.ndarray): Probabilities of each of the ``p`` different
            outcomes. The sum of these values must be 1.
        size (int or tuple of ints or None): Shape of a sample in each trial.
            For example when ``size`` is ``(a, b)``, shape of returned value is
            ``(a, b, p)`` where ``p`` is ``len(pvals)``.
            If ``size`` is ``None``, it is treated as ``()``. So, shape of
            returned value is ``(p,)``.

    Returns:
        cupy.ndarray: An array drawn from multinomial distribution.

    .. note::
       It does not support ``sum(pvals) < 1`` case.

    .. seealso:: :meth:`numpy.random.multinomial`
    