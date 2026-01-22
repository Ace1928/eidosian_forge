import itertools
from future.utils import with_metaclass
from .util import subvals
from .extend import (Box, primitive, notrace_primitive, VSpace, vspace,
class SequenceBox(Box):
    __slots__ = []
    __getitem__ = container_take

    def __len__(self):
        return len(self._value)

    def __add__(self, other):
        return sequence_extend_right(self, *other)

    def __radd__(self, other):
        return sequence_extend_left(self, *other)

    def __contains__(self, elt):
        return elt in self._value

    def index(self, elt):
        return self._value.index(elt)