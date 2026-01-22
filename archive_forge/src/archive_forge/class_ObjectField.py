class ObjectField(Node):
    __slots__ = ('loc', 'name', 'value')
    _fields = ('name', 'value')

    def __init__(self, name, value, loc=None):
        self.loc = loc
        self.name = name
        self.value = value

    def __eq__(self, other):
        return self is other or (isinstance(other, ObjectField) and self.name == other.name and (self.value == other.value))

    def __repr__(self):
        return 'ObjectField(name={self.name!r}, value={self.value!r})'.format(self=self)

    def __copy__(self):
        return type(self)(self.name, self.value, self.loc)

    def __hash__(self):
        return id(self)