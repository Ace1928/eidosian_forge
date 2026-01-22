from __future__ import absolute_import
import functools
import itertools
import operator
import sys
import types
def __get_module(self, fullname):
    try:
        return self.known_modules[fullname]
    except KeyError:
        raise ImportError('This loader does not know module ' + fullname)