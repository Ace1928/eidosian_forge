from __future__ import absolute_import, print_function, division
import io
import json
import inspect
from json.encoder import JSONEncoder
from os import unlink
from tempfile import NamedTemporaryFile
from petl.compat import PY2
from petl.compat import pickle
from petl.io.sources import read_source_from_arg, write_source_from_arg
from petl.util.base import data, Table, dicts as _dicts, iterpeek
class DictsGeneratorView(DictsView):

    def __init__(self, dicts, header=None, sample=1000, missing=None):
        super(DictsGeneratorView, self).__init__(dicts, header, sample, missing)
        self._filecache = None
        self._cached = 0

    def __iter__(self):
        if not self._header:
            self._determine_header()
        yield self._header
        if not self._filecache:
            if PY2:
                self._filecache = NamedTemporaryFile(delete=False, mode='wb+', bufsize=0)
            else:
                self._filecache = NamedTemporaryFile(delete=False, mode='wb+', buffering=0)
        position = 0
        it = iter(self.dicts)
        while True:
            if position < self._cached:
                self._filecache.seek(position)
                row = pickle.load(self._filecache)
                position = self._filecache.tell()
                yield row
                continue
            try:
                o = next(it)
            except StopIteration:
                break
            row = tuple((o.get(f, self.missing) for f in self._header))
            self._filecache.seek(self._cached)
            pickle.dump(row, self._filecache, protocol=-1)
            self._cached = position = self._filecache.tell()
            yield row

    def _determine_header(self):
        it = iter(self.dicts)
        header = list()
        peek, it = iterpeek(it, self.sample)
        self.dicts = it
        if isinstance(peek, dict):
            peek = [peek]
        for o in peek:
            if hasattr(o, 'keys'):
                header += [k for k in o.keys() if k not in header]
        self._header = tuple(header)
        return it

    def __del__(self):
        if self._filecache:
            self._filecache.close()
            unlink(self._filecache.name)