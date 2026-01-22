import sys
from breezy import tests
from breezy.tests import features
class _NoImplementCompare(_Hashable):

    def __eq__(self, other):
        return NotImplemented
    __hash__ = _Hashable.__hash__