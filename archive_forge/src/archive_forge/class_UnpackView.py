from __future__ import absolute_import, print_function, division
import itertools
from petl.compat import next, text_type
from petl.errors import ArgumentError
from petl.util.base import Table
class UnpackView(Table):

    def __init__(self, source, field, newfields=None, include_original=False, missing=None):
        self.source = source
        self.field = field
        self.newfields = newfields
        self.include_original = include_original
        self.missing = missing

    def __iter__(self):
        return iterunpack(self.source, self.field, self.newfields, self.include_original, self.missing)