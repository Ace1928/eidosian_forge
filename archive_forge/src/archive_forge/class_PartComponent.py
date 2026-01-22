important invariant is that the parts on the stack are themselves in
class PartComponent:
    """Internal class used in support of the multiset partitions
    enumerators and the associated visitor functions.

    Represents one component of one part of the current partition.

    A stack of these, plus an auxiliary frame array, f, represents a
    partition of the multiset.

    Knuth's pseudocode makes c, u, and v separate arrays.
    """
    __slots__ = ('c', 'u', 'v')

    def __init__(self):
        self.c = 0
        self.u = 0
        self.v = 0

    def __repr__(self):
        """for debug/algorithm animation purposes"""
        return 'c:%d u:%d v:%d' % (self.c, self.u, self.v)

    def __eq__(self, other):
        """Define  value oriented equality, which is useful for testers"""
        return isinstance(other, self.__class__) and self.c == other.c and (self.u == other.u) and (self.v == other.v)

    def __ne__(self, other):
        """Defined for consistency with __eq__"""
        return not self == other