class BooleanValue(Value):
    __slots__ = ('loc', 'value')
    _fields = ('value',)

    def __init__(self, value, loc=None):
        self.loc = loc
        self.value = value

    def __eq__(self, other):
        return self is other or (isinstance(other, BooleanValue) and self.value == other.value)

    def __repr__(self):
        return 'BooleanValue(value={self.value!r})'.format(self=self)

    def __copy__(self):
        return type(self)(self.value, self.loc)

    def __hash__(self):
        return id(self)