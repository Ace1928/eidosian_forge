import itertools
from future.utils import with_metaclass
from .util import subvals
from .extend import (Box, primitive, notrace_primitive, VSpace, vspace,
class DictVSpace(ContainerVSpace):

    def _values(self, x):
        return x.values()

    def _kv_pairs(self, x):
        return x.items()

    def _map(self, f, *args):
        return {k: f(vs, *[x[k] for x in args]) for k, vs in self.shape.items()}

    def _subval(self, xs, idx, x):
        d = dict(xs.items())
        d[idx] = x
        return d