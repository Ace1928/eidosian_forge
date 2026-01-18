from __future__ import absolute_import, print_function, division
from collections import Counter
from petl.compat import string_types, maketrans
from petl.util.base import values, Table, data, wrap
def valuecount(table, field, value, missing=None):
    """
    Count the number of occurrences of `value` under the given field. Returns
    the absolute count and relative frequency as a pair. E.g.::

        >>> import petl as etl
        >>> table = [['foo', 'bar'],
        ...          ['a', 1],
        ...          ['b', 2],
        ...          ['b', 7]]
        >>> etl.valuecount(table, 'foo', 'b')
        (2, 0.6666666666666666)

    The `field` argument can be a single field name or index (starting from
    zero) or a tuple of field names and/or indexes.

    """
    total = 0
    vs = 0
    for v in values(table, field, missing=missing):
        total += 1
        if v == value:
            vs += 1
    return (vs, float(vs) / total)