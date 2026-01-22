import copy
from .chemistry import Substance
from .units import (
from .util.arithmeticdict import ArithmeticDict, _imul, _itruediv
from .printing import as_per_substance_html_table
class AutoRegisteringSubstanceDict(object):

    def __init__(self, factory=Substance.from_formula):
        self.factory = factory
        self._store = {}

    def __getitem__(self, key):
        if key not in self._store:
            self._store[key] = self.factory(key)
        return self._store[key]