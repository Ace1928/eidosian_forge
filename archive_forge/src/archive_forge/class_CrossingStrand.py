import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
class CrossingStrand(BasicCrossingStrand):
    """
    One of the four strands of a crossing.
    """

    def rotate(self, s=1):
        """
        The CrossingStrand *counter-clockwise* from self.
        """
        c, e = (self.crossing, self.strand_index)
        return CrossingStrand(c, (e + s) % c._adjacent_len)

    def opposite(self):
        """
        The CrossingStrand at the other end of the edge from self
        """
        return CrossingStrand(*self.crossing.adjacent[self.strand_index])

    def next(self):
        """
        The CrossingStrand obtained by moving in the direction of self for
        one crossing
        """
        c, e = (self.crossing, self.strand_index)
        return CrossingStrand(*c.adjacent[(e + 2) % c._adjacent_len])

    def next_corner(self):
        c, e = (self.crossing, self.strand_index)
        return CrossingStrand(*c.adjacent[(e + 1) % c._adjacent_len])

    def previous_corner(self):
        return self.opposite().rotate(-1)

    def strand_label(self):
        return self.crossing.strand_labels[self.strand_index]

    def oriented(self):
        """
        Returns the one of {self, opposite} which is the *head* of the
        corresponding oriented edge of the link.
        """
        c, e = (self.crossing, self.strand_index)
        if c.sign == 1 and e in [0, 3] or (c.sign == -1 and e in [0, 1]):
            return self
        return self.opposite()

    def __repr__(self):
        return '<CS %s, %s>' % (self.crossing, self.strand_index)