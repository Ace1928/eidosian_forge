from __future__ import absolute_import, print_function, division
import operator
from petl.compat import string_types, izip
from petl.errors import ArgumentError
from petl.util.base import Table, dicts
def totextindex(table, index_or_dirname, schema=None, indexname=None, merge=False, optimize=False):
    """
    Load all rows from `table` into a Whoosh index. N.B., this will clear any
    existing data in the index before loading. E.g.::

        >>> import petl as etl
        >>> import datetime
        >>> import os
        >>> # here is the table we want to load into an index
        ... table = (('f0', 'f1', 'f2', 'f3', 'f4'),
        ...          ('AAA', 12, 4.3, True, datetime.datetime.now()),
        ...          ('BBB', 6, 3.4, False, datetime.datetime(1900, 1, 31)),
        ...          ('CCC', 42, 7.8, True, datetime.datetime(2100, 12, 25)))
        >>> # define a schema for the index
        ... from whoosh.fields import *
        >>> schema = Schema(f0=TEXT(stored=True),
        ...                 f1=NUMERIC(int, stored=True),
        ...                 f2=NUMERIC(float, stored=True),
        ...                 f3=BOOLEAN(stored=True),
        ...                 f4=DATETIME(stored=True))
        >>> # load index
        ... dirname = 'example.whoosh'
        >>> if not os.path.exists(dirname):
        ...     os.mkdir(dirname)
        ...
        >>> etl.totextindex(table, dirname, schema=schema)

    Keyword arguments:

    table
        A table container with the data to be loaded.
    index_or_dirname
        Either an instance of `whoosh.index.Index` or a string containing the
        directory path where the index is to be stored.
    schema
        Index schema to use if creating the index.
    indexname
        String containing the name of the index, if multiple indexes are stored
        in the same directory.
    merge
        Merge small segments during commit?
    optimize
        Merge all segments together?

    """
    import whoosh.index
    import whoosh.writing
    if isinstance(index_or_dirname, string_types):
        dirname = index_or_dirname
        index = whoosh.index.create_in(dirname, schema, indexname=indexname)
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
        writer.commit(merge=merge, optimize=optimize, mergetype=whoosh.writing.CLEAR)
    except:
        writer.cancel()
        raise
    finally:
        if needs_closing:
            index.close()