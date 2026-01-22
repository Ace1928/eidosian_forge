import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
class Strand:
    """
    When constructing links, it's convenient to have strands as well
    as crossings.  These are stripped by the Link class when it
    pieces things together.
    """

    def __init__(self, label=None, component_idx=None):
        self.label = label
        self.adjacent = [None, None]
        self._adjacent_len = 2
        self.component_idx = component_idx

    def fuse(self):
        """
        Joins the incoming and outgoing strands and removes
        self from the picture.
        """
        (a, i), (b, j) = self.adjacent
        a.adjacent[i] = (b, j)
        b.adjacent[j] = (a, i)

    def __getitem__(self, i):
        return (self, i % 2)

    def __setitem__(self, i, other):
        o, j = other
        self.adjacent[i % 2] = other
        other[0].adjacent[other[1]] = (self, i)

    def __repr__(self):
        return '%s' % self.label

    def info(self):

        def format_adjacent(a):
            return (a[0].label, a[1]) if a else None
        print('<%s : %s>' % (self.label, [format_adjacent(a) for a in self.adjacent]))

    def is_loop(self):
        return self == self.adjacent[0][0]