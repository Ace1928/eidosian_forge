from __future__ import absolute_import, print_function, division
from collections import Counter
from petl.compat import string_types, maketrans
from petl.util.base import values, Table, data, wrap
def valuecounter(table, *field, **kwargs):
    """
    Find distinct values for the given field and count the number of
    occurrences. Returns a :class:`dict` mapping values to counts. E.g.::

        >>> import petl as etl
        >>> table = [['foo', 'bar'],
        ...          ['a', True],
        ...          ['b'],
        ...          ['b', True],
        ...          ['c', False]]
        >>> etl.valuecounter(table, 'foo')
        Counter({'b': 2, 'a': 1, 'c': 1})

    The `field` argument can be a single field name or index (starting from
    zero) or a tuple of field names and/or indexes.

    """
    missing = kwargs.get('missing', None)
    counter = Counter()
    for v in values(table, field, missing=missing):
        try:
            counter[v] += 1
        except IndexError:
            pass
    return counter