import itertools
from future.utils import with_metaclass
from .util import subvals
from .extend import (Box, primitive, notrace_primitive, VSpace, vspace,
class list(with_metaclass(ListMeta, list_)):

    def __new__(cls, xs):
        return make_sequence(list_, *xs)