from __future__ import absolute_import, print_function, division
import itertools
from petl.compat import string_types, text_type
from petl.util.base import Table, iterpeek
from petl.io.numpy import construct_dtype
def tobcolz(table, dtype=None, sample=1000, **kwargs):
    """Load data into a bcolz ctable, e.g.::

        >>> import petl as etl
        >>>
        >>> def example_to_bcolz():
        ...     table = [('foo', 'bar', 'baz'),
        ...              ('apples', 1, 2.5),
        ...              ('oranges', 3, 4.4),
        ...              ('pears', 7, .1)]
        ...     return etl.tobcolz(table)
        >>> 
        >>> ctbl = example_to_bcolz() # doctest: +SKIP
        >>> ctbl # doctest: +SKIP
        ctable((3,), [('foo', '<U7'), ('bar', '<i8'), ('baz', '<f8')])
          nbytes: 132; cbytes: 1023.98 KB; ratio: 0.00
          cparams := cparams(clevel=5, shuffle=1, cname='lz4', quantize=0)
        [('apples', 1, 2.5) ('oranges', 3, 4.4) ('pears', 7, 0.1)]
        >>> ctbl.names # doctest: +SKIP
        ['foo', 'bar', 'baz']
        >>> ctbl['foo'] # doctest: +SKIP
        carray((3,), <U7)
          nbytes := 84; cbytes := 511.98 KB; ratio: 0.00
          cparams := cparams(clevel=5, shuffle=1, cname='lz4', quantize=0)
          chunklen := 18724; chunksize: 524272; blocksize: 0
        ['apples' 'oranges' 'pears']

    Other keyword arguments are passed through to the ctable constructor.

    .. versionadded:: 1.1.0

    """
    import bcolz
    import numpy as np
    it = iter(table)
    peek, it = iterpeek(it, sample)
    hdr = next(it)
    it = (tuple(row) for row in it)
    flds = list(map(text_type, hdr))
    dtype = construct_dtype(flds, peek, dtype)
    kwargs.setdefault('expectedlen', 1000000)
    kwargs.setdefault('mode', 'w')
    ctbl = bcolz.ctable(np.array([], dtype=dtype), **kwargs)
    chunklen = sum((ctbl.cols[name].chunklen for name in ctbl.names)) // len(ctbl.names)
    while True:
        data = list(itertools.islice(it, chunklen))
        data = np.array(data, dtype=dtype)
        ctbl.append(data)
        if len(data) < chunklen:
            break
    ctbl.flush()
    return ctbl