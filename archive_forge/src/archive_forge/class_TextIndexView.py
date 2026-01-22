from __future__ import absolute_import, print_function, division
import operator
from petl.compat import string_types, izip
from petl.errors import ArgumentError
from petl.util.base import Table, dicts
class TextIndexView(Table):

    def __init__(self, index_or_dirname, indexname=None, docnum_field=None):
        self.index_or_dirname = index_or_dirname
        self.indexname = indexname
        self.docnum_field = docnum_field

    def __iter__(self):
        return itertextindex(self.index_or_dirname, self.indexname, self.docnum_field)