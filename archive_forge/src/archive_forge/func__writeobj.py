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
def _writeobj(encoder, obj, f, prefix, suffix, lines=False):
    if prefix is not None:
        f.write(prefix)
    if lines:
        for rec in obj:
            for chunk in encoder.iterencode(rec):
                f.write(chunk)
            f.write('\n')
    else:
        for chunk in encoder.iterencode(obj):
            f.write(chunk)
    if suffix is not None:
        f.write(suffix)