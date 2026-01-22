import itertools
from future.utils import with_metaclass
from .util import subvals
from .extend import (Box, primitive, notrace_primitive, VSpace, vspace,
class NamedTupleVSpace(SequenceVSpace):

    def _map(self, f, *args):
        return self.seq_type(*map(f, self.shape, *args))

    def _subval(self, xs, idx, x):
        return self.seq_type(*subvals(xs, [(idx, x)]))