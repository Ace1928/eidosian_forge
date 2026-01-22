from __future__ import absolute_import, print_function, division
import io
from petl.compat import next, PY2, text_type
from petl.util.base import Table, asdict
from petl.io.base import getcodec
from petl.io.sources import read_source_from_arg, write_source_from_arg
class TeeTextView(Table):

    def __init__(self, table, source=None, encoding=None, errors='strict', template=None, prologue=None, epilogue=None):
        self.table = table
        self.source = source
        self.encoding = encoding
        self.errors = errors
        self.template = template
        self.prologue = prologue
        self.epilogue = epilogue

    def __iter__(self):
        return _iterteetext(self.table, self.source, self.encoding, self.errors, self.template, self.prologue, self.epilogue)