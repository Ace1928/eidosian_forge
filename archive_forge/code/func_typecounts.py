from __future__ import absolute_import, print_function, division
from collections import Counter
from petl.compat import string_types, maketrans
from petl.util.base import values, Table, data, wrap
def typecounts(table, field):
    """
    Count the number of values found for each Python type and return a table
    mapping class names to counts and frequencies. E.g.::

        >>> import petl as etl
        >>> table = [['foo', 'bar', 'baz'],
        ...          [b'A', 1, 2],
        ...          [b'B', '2', b'3.4'],
        ...          ['B', '3', '7.8', True],
        ...          ['D', u'xyz', 9.0],
        ...          ['E', 42]]
        >>> etl.typecounts(table, 'foo')
        +---------+-------+-----------+
        | type    | count | frequency |
        +=========+=======+===========+
        | 'str'   |     3 |       0.6 |
        +---------+-------+-----------+
        | 'bytes' |     2 |       0.4 |
        +---------+-------+-----------+

        >>> etl.typecounts(table, 'bar')
        +-------+-------+-----------+
        | type  | count | frequency |
        +=======+=======+===========+
        | 'str' |     3 |       0.6 |
        +-------+-------+-----------+
        | 'int' |     2 |       0.4 |
        +-------+-------+-----------+

        >>> etl.typecounts(table, 'baz')
        +------------+-------+-----------+
        | type       | count | frequency |
        +============+=======+===========+
        | 'int'      |     1 |       0.2 |
        +------------+-------+-----------+
        | 'bytes'    |     1 |       0.2 |
        +------------+-------+-----------+
        | 'str'      |     1 |       0.2 |
        +------------+-------+-----------+
        | 'float'    |     1 |       0.2 |
        +------------+-------+-----------+
        | 'NoneType' |     1 |       0.2 |
        +------------+-------+-----------+

    The `field` argument can be a field name or index (starting from zero).

    """
    return TypeCountsView(table, field)