from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
def addrownumbers(table, start=1, step=1, field='row'):
    """
    Add a field of row numbers. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['A', 9],
        ...           ['C', 2],
        ...           ['F', 1]]
        >>> table2 = etl.addrownumbers(table1)
        >>> table2
        +-----+-----+-----+
        | row | foo | bar |
        +=====+=====+=====+
        |   1 | 'A' |   9 |
        +-----+-----+-----+
        |   2 | 'C' |   2 |
        +-----+-----+-----+
        |   3 | 'F' |   1 |
        +-----+-----+-----+

    Parameters `start` and `step` control the numbering.

    """
    return AddRowNumbersView(table, start, step, field)