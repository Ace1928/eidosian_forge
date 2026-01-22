class InputValueDefinition(Node):
    __slots__ = ('loc', 'name', 'type', 'default_value', 'directives')
    _fields = ('name', 'type', 'default_value')

    def __init__(self, name, type, default_value=None, loc=None, directives=None):
        self.loc = loc
        self.name = name
        self.type = type
        self.default_value = default_value
        self.directives = directives

    def __eq__(self, other):
        return self is other or (isinstance(other, InputValueDefinition) and self.name == other.name and (self.type == other.type) and (self.default_value == other.default_value) and (self.directives == other.directives))

    def __repr__(self):
        return 'InputValueDefinition(name={self.name!r}, type={self.type!r}, default_value={self.default_value!r}, directives={self.directives!r})'.format(self=self)

    def __copy__(self):
        return type(self)(self.name, self.type, self.default_value, self.loc, self.directives)

    def __hash__(self):
        return id(self)