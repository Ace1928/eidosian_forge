from __future__ import absolute_import, print_function, division
import logging
from petl.compat import callable
def _hasmethods(o, *l):
    return all((_hasmethod(o, n) for n in l))