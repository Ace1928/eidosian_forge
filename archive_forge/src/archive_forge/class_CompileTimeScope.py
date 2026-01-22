from __future__ import absolute_import
import cython
import os
import platform
from unicodedata import normalize
from contextlib import contextmanager
from .. import Utils
from ..Plex.Scanners import Scanner
from ..Plex.Errors import UnrecognizedInput
from .Errors import error, warning, hold_errors, release_errors, CompileError
from .Lexicon import any_string_prefix, make_lexicon, IDENT
from .Future import print_function
class CompileTimeScope(object):

    def __init__(self, outer=None):
        self.entries = {}
        self.outer = outer

    def declare(self, name, value):
        self.entries[name] = value

    def update(self, other):
        self.entries.update(other)

    def lookup_here(self, name):
        return self.entries[name]

    def __contains__(self, name):
        return name in self.entries

    def lookup(self, name):
        try:
            return self.lookup_here(name)
        except KeyError:
            outer = self.outer
            if outer:
                return outer.lookup(name)
            else:
                raise