from __future__ import absolute_import, print_function, division
import os
import heapq
from tempfile import NamedTemporaryFile
import itertools
import logging
from collections import namedtuple
import operator
from petl.compat import pickle, next, text_type
import petl.config as config
from petl.comparison import comparable_itemgetter
from petl.util.base import Table, asindices
def _iternocache(self, source, key, reverse):
    debug('iterate without cache')
    self.clearcache()
    it = iter(source)
    try:
        hdr = next(it)
    except StopIteration:
        if key is None:
            return
        hdr = []
    yield tuple(hdr)
    if key is not None:
        indices = asindices(hdr, key)
    else:
        indices = range(len(hdr))
    getkey = comparable_itemgetter(*indices)
    rows = list(itertools.islice(it, 0, self.buffersize))
    rows.sort(key=getkey, reverse=reverse)
    if self.buffersize is None or len(rows) < self.buffersize:
        if self.cache:
            debug('caching mem')
            self._hdrcache = hdr
            self._memcache = rows
            self._getkey = getkey
        for row in rows:
            yield tuple(row)
    else:
        chunkfiles = []
        while rows:
            with NamedTemporaryFile(dir=self.tempdir, delete=False, mode='wb') as f:
                wrapper = _NamedTempFileDeleteOnGC(f.name)
                debug('created temporary chunk file %s' % f.name)
                for row in rows:
                    pickle.dump(row, f, protocol=-1)
                f.flush()
                chunkfiles.append(wrapper)
            rows = list(itertools.islice(it, 0, self.buffersize))
            rows.sort(key=getkey, reverse=reverse)
        if self.cache:
            debug('caching files')
            self._hdrcache = hdr
            self._filecache = chunkfiles
            self._getkey = getkey
        chunkiters = [_iterchunk(f.name) for f in chunkfiles]
        for row in _mergesorted(getkey, reverse, *chunkiters):
            yield tuple(row)