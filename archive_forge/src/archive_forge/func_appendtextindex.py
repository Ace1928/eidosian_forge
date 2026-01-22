from __future__ import absolute_import, print_function, division
import operator
from petl.compat import string_types, izip
from petl.errors import ArgumentError
from petl.util.base import Table, dicts
def appendtextindex(table, index_or_dirname, indexname=None, merge=True, optimize=False):
    """
    Load all rows from `table` into a Whoosh index, adding them to any existing
    data in the index.

    Keyword arguments:

    table
        A table container with the data to be loaded.
    index_or_dirname
        Either an instance of `whoosh.index.Index` or a string containing the
        directory path where the index is to be stored.
    indexname
        String containing the name of the index, if multiple indexes are stored
        in the same directory.
    merge
        Merge small segments during commit?
    optimize
        Merge all segments together?

    """
    import whoosh.index
    if isinstance(index_or_dirname, string_types):
        dirname = index_or_dirname
        index = whoosh.index.open_dir(dirname, indexname=indexname, readonly=False)
        needs_closing = True
    elif isinstance(index_or_dirname, whoosh.index.Index):
        index = index_or_dirname
        needs_closing = False
    else:
        raise ArgumentError('expected string or index, found %r' % index_or_dirname)
    writer = index.writer()
    try:
        for d in dicts(table):
            writer.add_document(**d)
        writer.commit(merge=merge, optimize=optimize)
    except Exception:
        writer.cancel()
        raise
    finally:
        if needs_closing:
            index.close()