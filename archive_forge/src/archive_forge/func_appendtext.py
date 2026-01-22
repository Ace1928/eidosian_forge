from __future__ import absolute_import, print_function, division
import io
from petl.compat import next, PY2, text_type
from petl.util.base import Table, asdict
from petl.io.base import getcodec
from petl.io.sources import read_source_from_arg, write_source_from_arg
def appendtext(table, source=None, encoding=None, errors='strict', template=None, prologue=None, epilogue=None):
    """
    Append the table to a text file.

    """
    _writetext(table, source=source, mode='ab', encoding=encoding, errors=errors, template=template, prologue=prologue, epilogue=epilogue)