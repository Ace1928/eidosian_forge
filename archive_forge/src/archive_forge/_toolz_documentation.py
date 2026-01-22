from toolz as it appeared when copying the code.
import operator
from functools import reduce

    NB: This is a straight copy of the get_in implementation found in
        the toolz library (https://github.com/pytoolz/toolz/). It works
        with persistent data structures as well as the corresponding
        datastructures from the stdlib.

    Returns coll[i0][i1]...[iX] where [i0, i1, ..., iX]==keys.

    If coll[i0][i1]...[iX] cannot be found, returns ``default``, unless
    ``no_default`` is specified, then it raises KeyError or IndexError.

    ``get_in`` is a generalization of ``operator.getitem`` for nested data
    structures such as dictionaries and lists.
    >>> from pyrsistent import freeze
    >>> transaction = freeze({'name': 'Alice',
    ...                       'purchase': {'items': ['Apple', 'Orange'],
    ...                                    'costs': [0.50, 1.25]},
    ...                       'credit card': '5555-1234-1234-1234'})
    >>> get_in(['purchase', 'items', 0], transaction)
    'Apple'
    >>> get_in(['name'], transaction)
    'Alice'
    >>> get_in(['purchase', 'total'], transaction)
    >>> get_in(['purchase', 'items', 'apple'], transaction)
    >>> get_in(['purchase', 'items', 10], transaction)
    >>> get_in(['purchase', 'total'], transaction, 0)
    0
    >>> get_in(['y'], {}, no_default=True)
    Traceback (most recent call last):
    ...
    KeyError: 'y'
    