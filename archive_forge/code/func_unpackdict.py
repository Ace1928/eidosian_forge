from __future__ import absolute_import, print_function, division
import itertools
from petl.compat import next, text_type
from petl.errors import ArgumentError
from petl.util.base import Table
def unpackdict(table, field, keys=None, includeoriginal=False, samplesize=1000, missing=None):
    """
    Unpack dictionary values into separate fields. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           [1, {'baz': 'a', 'quux': 'b'}],
        ...           [2, {'baz': 'c', 'quux': 'd'}],
        ...           [3, {'baz': 'e', 'quux': 'f'}]]
        >>> table2 = etl.unpackdict(table1, 'bar')
        >>> table2
        +-----+-----+------+
        | foo | baz | quux |
        +=====+=====+======+
        |   1 | 'a' | 'b'  |
        +-----+-----+------+
        |   2 | 'c' | 'd'  |
        +-----+-----+------+
        |   3 | 'e' | 'f'  |
        +-----+-----+------+

    See also :func:`petl.transform.unpacks.unpack`.

    """
    return UnpackDictView(table, field, keys=keys, includeoriginal=includeoriginal, samplesize=samplesize, missing=missing)