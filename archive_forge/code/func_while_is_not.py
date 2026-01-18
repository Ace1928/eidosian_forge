from collections import abc
import functools
import itertools
@_ensure_iterable
def while_is_not(it, stop_value):
    """Yields given values from iterator until stop value is passed.

    This uses the ``is`` operator to determine equivalency (and not the
    ``==`` operator).
    """
    for value in it:
        yield value
        if value is stop_value:
            break