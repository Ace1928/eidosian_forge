from __future__ import print_function
import collections
import inspect
import itertools
import operator
import re
import sys
class getfullargspec(object):
    """A quick and dirty replacement for getfullargspec for Python 2.X"""

    def __init__(self, f):
        self.args, self.varargs, self.varkw, self.defaults = inspect.getargspec(f)
        self.kwonlyargs = []
        self.kwonlydefaults = None

    def __iter__(self):
        yield self.args
        yield self.varargs
        yield self.varkw
        yield self.defaults
    getargspec = inspect.getargspec