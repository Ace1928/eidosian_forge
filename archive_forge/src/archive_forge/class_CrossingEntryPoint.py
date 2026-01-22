import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
class CrossingEntryPoint(CrossingStrand):
    """
    One of the two entry points of an oriented crossing
    """

    def next(self):
        c, e = (self.crossing, self.strand_index)
        s = c._adjacent_len // 2
        return CrossingEntryPoint(*c.adjacent[(e + s) % (2 * s)])

    def previous(self):
        s = self.crossing._adjacent_len // 2
        return CrossingEntryPoint(*self.opposite().rotate(s))

    def other(self):
        nonzero_entry_point = 1 if self.crossing.sign == -1 else 3
        other = nonzero_entry_point if self.strand_index == 0 else 0
        return CrossingEntryPoint(self.crossing, other)

    def is_under_crossing(self):
        return self.strand_index == 0

    def is_over_crossing(self):
        return self.strand_index != 0

    def component(self):
        ans = [self]
        while True:
            next = ans[-1].next()
            if next == self:
                break
            else:
                ans.append(next)
        return ans

    def component_label(self):
        return self.crossing.strand_components[self.strand_index]

    def label_crossing(self, comp, labels):
        c, e = (self.crossing, self.strand_index)
        f = (e + 2) % 4
        c.strand_labels[e], c.strand_components[e] = (labels[self], comp)
        c.strand_labels[f], c.strand_components[f] = (labels[self.next()], comp)

    def __repr__(self):
        return '<CEP %s, %s>' % (self.crossing, self.strand_index)