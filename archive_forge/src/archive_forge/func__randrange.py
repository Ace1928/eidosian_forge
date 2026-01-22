import ``rng`` and access the method directly. For example, to capture the
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int
import random as _random
def _randrange(seed=None):
    """Return a randrange generator.

    ``seed`` can be

    * None - return randomly seeded generator
    * int - return a generator seeded with the int
    * list - the values to be returned will be taken from the list
      in the order given; the provided list is not modified.

    Examples
    ========

    >>> from sympy.core.random import _randrange
    >>> rr = _randrange()
    >>> rr(1000) # doctest: +SKIP
    999
    >>> rr = _randrange(3)
    >>> rr(1000) # doctest: +SKIP
    238
    >>> rr = _randrange([0, 5, 1, 3, 4])
    >>> rr(3), rr(3)
    (0, 1)
    """
    if seed is None:
        return randrange
    elif isinstance(seed, int):
        rng.seed(seed)
        return randrange
    elif is_sequence(seed):
        seed = list(seed)
        seed.reverse()

        def give(a, b=None, seq=seed):
            if b is None:
                a, b = (0, a)
            a, b = (as_int(a), as_int(b))
            w = b - a
            if w < 1:
                raise ValueError('_randrange got empty range')
            try:
                x = seq.pop()
            except IndexError:
                raise ValueError('_randrange sequence was too short')
            if a <= x < b:
                return x
            else:
                return give(a, b, seq)
        return give
    else:
        raise ValueError('_randrange got an unexpected seed')