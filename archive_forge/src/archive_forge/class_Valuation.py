import inspect
import re
import sys
import textwrap
from pprint import pformat
from nltk.decorators import decorator  # this used in code that is commented out
from nltk.sem.logic import (
class Valuation(dict):
    """
    A dictionary which represents a model-theoretic Valuation of non-logical constants.
    Keys are strings representing the constants to be interpreted, and values correspond
    to individuals (represented as strings) and n-ary relations (represented as sets of tuples
    of strings).

    An instance of ``Valuation`` will raise a KeyError exception (i.e.,
    just behave like a standard  dictionary) if indexed with an expression that
    is not in its list of symbols.
    """

    def __init__(self, xs):
        """
        :param xs: a list of (symbol, value) pairs.
        """
        super().__init__()
        for sym, val in xs:
            if isinstance(val, str) or isinstance(val, bool):
                self[sym] = val
            elif isinstance(val, set):
                self[sym] = set2rel(val)
            else:
                msg = textwrap.fill("Error in initializing Valuation. Unrecognized value for symbol '%s':\n%s" % (sym, val), width=66)
                raise ValueError(msg)

    def __getitem__(self, key):
        if key in self:
            return dict.__getitem__(self, key)
        else:
            raise Undefined("Unknown expression: '%s'" % key)

    def __str__(self):
        return pformat(self)

    @property
    def domain(self):
        """Set-theoretic domain of the value-space of a Valuation."""
        dom = []
        for val in self.values():
            if isinstance(val, str):
                dom.append(val)
            elif not isinstance(val, bool):
                dom.extend([elem for tuple_ in val for elem in tuple_ if elem is not None])
        return set(dom)

    @property
    def symbols(self):
        """The non-logical constants which the Valuation recognizes."""
        return sorted(self.keys())

    @classmethod
    def fromstring(cls, s):
        return read_valuation(s)