from __future__ import absolute_import, print_function, division
from contextlib import contextmanager
from petl.compat import string_types
from petl.errors import ArgumentError
from petl.util.base import Table, iterpeek, data
from petl.io.numpy import infer_dtype
@contextmanager
def _get_hdf5_table(source, where, name, mode='r'):
    import tables
    needs_closing = False
    h5file = None
    if isinstance(source, tables.Table):
        h5tbl = source
    elif isinstance(source, string_types):
        h5file = tables.open_file(source, mode=mode)
        needs_closing = True
        h5tbl = h5file.get_node(where, name=name)
    elif isinstance(source, tables.File):
        h5file = source
        h5tbl = h5file.get_node(where, name=name)
    else:
        raise ArgumentError('invalid source argument, expected file name or tables.File or tables.Table object, found: %r' % source)
    try:
        yield h5tbl
    finally:
        if needs_closing:
            h5file.close()