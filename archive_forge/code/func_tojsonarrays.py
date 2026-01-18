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
def tojsonarrays(table, source=None, prefix=None, suffix=None, output_header=False, *args, **kwargs):
    """
    Write a table in JSON format, with rows output as JSON arrays. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['a', 1],
        ...           ['b', 2],
        ...           ['c', 2]]
        >>> etl.tojsonarrays(table1, 'example.file4.json')
        >>> # check what it did
        ... print(open('example.file4.json').read())
        [["a", 1], ["b", 2], ["c", 2]]

    Note that this is currently not streaming, all data is loaded into memory
    before being written to the file.

    """
    if output_header:
        obj = list(table)
    else:
        obj = list(data(table))
    _writejson(source, obj, prefix, suffix, *args, **kwargs)