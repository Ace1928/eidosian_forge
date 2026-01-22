from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import sys
import gast as ast
class ordered_set(object):

    def __init__(self, elements=None):
        self.values = OrderedDict.fromkeys(elements or [])

    def add(self, value):
        self.values[value] = None

    def update(self, values):
        self.values.update(((k, None) for k in values))

    def __iter__(self):
        return iter(self.values.keys())

    def __contains__(self, value):
        return value in self.values

    def __add__(self, other):
        out = self.values.copy()
        out.update(other.values)
        return out

    def __len__(self):
        return len(self.values)