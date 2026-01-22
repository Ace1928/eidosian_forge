from cupy.random import _generator
Returns a permuted range or a permutation of an array.

    Args:
        a (int or cupy.ndarray): The range or the array to be shuffled.

    Returns:
        cupy.ndarray: If `a` is an integer, it is permutation range between 0
        and `a` - 1.
        Otherwise, it is a permutation of `a`.

    .. seealso:: :meth:`numpy.random.permutation`
    