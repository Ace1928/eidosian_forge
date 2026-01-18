from __future__ import division, print_function, absolute_import
import inspect
from petl.util.base import Table
def todataframe(table, index=None, exclude=None, columns=None, coerce_float=False, nrows=None):
    """
    Load data from the given `table` into a
    `pandas <http://pandas.pydata.org/>`_ DataFrame. E.g.::

        >>> import petl as etl
        >>> table = [('foo', 'bar', 'baz'),
        ...          ('apples', 1, 2.5),
        ...          ('oranges', 3, 4.4),
        ...          ('pears', 7, .1)]
        >>> df = etl.todataframe(table)
        >>> df
               foo  bar  baz
        0   apples    1  2.5
        1  oranges    3  4.4
        2    pears    7  0.1

    """
    import pandas as pd
    it = iter(table)
    try:
        header = next(it)
    except StopIteration:
        header = None
    if columns is None:
        columns = header
    return pd.DataFrame.from_records(it, index=index, exclude=exclude, columns=columns, coerce_float=coerce_float, nrows=nrows)