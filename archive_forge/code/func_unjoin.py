from __future__ import absolute_import, print_function, division
import itertools
import operator
from petl.compat import next, text_type
from petl.errors import ArgumentError
from petl.comparison import comparable_itemgetter, Comparable
from petl.util.base import Table, asindices, rowgetter, rowgroupby, \
from petl.transform.sorts import sort
from petl.transform.basics import cut, cutout
from petl.transform.dedup import distinct
def unjoin(table, value, key=None, autoincrement=(1, 1), presorted=False, buffersize=None, tempdir=None, cache=True):
    """
    Split a table into two tables by reversing an inner join. E.g.::

        >>> import petl as etl
        >>> # join key is present in the table
        ... table1 = (('foo', 'bar', 'baz'),
        ...           ('A', 1, 'apple'),
        ...           ('B', 1, 'apple'),
        ...           ('C', 2, 'orange'))
        >>> table2, table3 = etl.unjoin(table1, 'baz', key='bar')
        >>> table2
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'A' |   1 |
        +-----+-----+
        | 'B' |   1 |
        +-----+-----+
        | 'C' |   2 |
        +-----+-----+

        >>> table3
        +-----+----------+
        | bar | baz      |
        +=====+==========+
        |   1 | 'apple'  |
        +-----+----------+
        |   2 | 'orange' |
        +-----+----------+

        >>> # an integer join key can also be reconstructed
        ... table4 = (('foo', 'bar'),
        ...           ('A', 'apple'),
        ...           ('B', 'apple'),
        ...           ('C', 'orange'))
        >>> table5, table6 = etl.unjoin(table4, 'bar')
        >>> table5
        +-----+--------+
        | foo | bar_id |
        +=====+========+
        | 'A' |      1 |
        +-----+--------+
        | 'B' |      1 |
        +-----+--------+
        | 'C' |      2 |
        +-----+--------+

        >>> table6
        +----+----------+
        | id | bar      |
        +====+==========+
        |  1 | 'apple'  |
        +----+----------+
        |  2 | 'orange' |
        +----+----------+

    The `autoincrement` parameter controls how an integer join key is
    reconstructed, and should be a tuple of (`start`, `step`).

    """
    if key is None:
        if presorted:
            tbl_sorted = table
        else:
            tbl_sorted = sort(table, value, buffersize=buffersize, tempdir=tempdir, cache=cache)
        left = ConvertToIncrementingCounterView(tbl_sorted, value, autoincrement)
        right = EnumerateDistinctView(tbl_sorted, value, autoincrement)
    else:
        left = distinct(cutout(table, value))
        right = distinct(cut(table, key, value))
    return (left, right)