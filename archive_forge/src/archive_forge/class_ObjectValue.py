class ObjectValue(Value):
    __slots__ = ('loc', 'fields')
    _fields = ('fields',)

    def __init__(self, fields, loc=None):
        self.loc = loc
        self.fields = fields

    def __eq__(self, other):
        return self is other or (isinstance(other, ObjectValue) and self.fields == other.fields)

    def __repr__(self):
        return 'ObjectValue(fields={self.fields!r})'.format(self=self)

    def __copy__(self):
        return type(self)(self.fields, self.loc)

    def __hash__(self):
        return id(self)