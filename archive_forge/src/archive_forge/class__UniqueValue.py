from __future__ import print_function
from patsy import PatsyError
from patsy.origin import Origin
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
class _UniqueValue(object):

    def __init__(self, print_as):
        self._print_as = print_as

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self._print_as)
    __getstate__ = no_pickling