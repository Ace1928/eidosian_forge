from __future__ import absolute_import, print_function, division
from petl.compat import next
from petl.util.base import Table, asindices

    Replace missing values with following non-missing values. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar', 'baz'],
        ...           [1, 'a', None],
        ...           [1, None, .23],
        ...           [1, 'b', None],
        ...           [2, None, None],
        ...           [2, None, .56],
        ...           [2, 'c', None],
        ...           [None, 'c', .72]]
        >>> table2 = etl.fillleft(table1)
        >>> table2.lookall()
        +-----+------+------+
        | foo | bar  | baz  |
        +=====+======+======+
        |   1 | 'a'  | None |
        +-----+------+------+
        |   1 | 0.23 | 0.23 |
        +-----+------+------+
        |   1 | 'b'  | None |
        +-----+------+------+
        |   2 | None | None |
        +-----+------+------+
        |   2 | 0.56 | 0.56 |
        +-----+------+------+
        |   2 | 'c'  | None |
        +-----+------+------+
        | 'c' | 'c'  | 0.72 |
        +-----+------+------+

    Use the `missing` keyword argument to control which value is treated as
    missing (`None` by default).

    