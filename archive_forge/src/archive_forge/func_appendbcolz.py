from __future__ import absolute_import, print_function, division
import itertools
from petl.compat import string_types, text_type
from petl.util.base import Table, iterpeek
from petl.io.numpy import construct_dtype
def appendbcolz(table, obj, check_names=True):
    """Append data into a bcolz ctable. The `obj` argument can be either an
    existing ctable or the name of a directory were an on-disk ctable is
    stored.

    .. versionadded:: 1.1.0

    """
    import bcolz
    import numpy as np
    if isinstance(obj, string_types):
        ctbl = bcolz.open(obj, mode='a')
    else:
        assert hasattr(obj, 'append') and hasattr(obj, 'names'), 'expected rootdir or ctable, found %r' % obj
        ctbl = obj
    dtype = ctbl.dtype
    it = iter(table)
    hdr = next(it)
    flds = list(map(text_type, hdr))
    if check_names:
        assert tuple(flds) == tuple(ctbl.names), 'column names do not match'
    chunklen = sum((ctbl.cols[name].chunklen for name in ctbl.names)) // len(ctbl.names)
    while True:
        data = list(itertools.islice(it, chunklen))
        data = np.array(data, dtype=dtype)
        ctbl.append(data)
        if len(data) < chunklen:
            break
    ctbl.flush()
    return ctbl