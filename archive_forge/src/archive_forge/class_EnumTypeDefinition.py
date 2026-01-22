class EnumTypeDefinition(TypeDefinition):
    __slots__ = ('loc', 'name', 'values', 'directives')
    _fields = ('name', 'values')

    def __init__(self, name, values, loc=None, directives=None):
        self.loc = loc
        self.name = name
        self.values = values
        self.directives = directives

    def __eq__(self, other):
        return self is other or (isinstance(other, EnumTypeDefinition) and self.name == other.name and (self.values == other.values) and (self.directives == other.directives))

    def __repr__(self):
        return 'EnumTypeDefinition(name={self.name!r}, values={self.values!r}, directives={self.directives!r})'.format(self=self)

    def __copy__(self):
        return type(self)(self.name, self.values, self.loc, self.directives)

    def __hash__(self):
        return id(self)