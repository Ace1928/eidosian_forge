class SelectionSet(Node):
    __slots__ = ('loc', 'selections')
    _fields = ('selections',)

    def __init__(self, selections, loc=None):
        self.loc = loc
        self.selections = selections

    def __eq__(self, other):
        return self is other or (isinstance(other, SelectionSet) and self.selections == other.selections)

    def __repr__(self):
        return 'SelectionSet(selections={self.selections!r})'.format(self=self)

    def __copy__(self):
        return type(self)(self.selections, self.loc)

    def __hash__(self):
        return id(self)