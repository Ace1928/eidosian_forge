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