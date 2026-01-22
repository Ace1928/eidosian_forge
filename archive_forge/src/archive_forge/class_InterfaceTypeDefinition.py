class InterfaceTypeDefinition(TypeDefinition):
    __slots__ = ('loc', 'name', 'fields', 'directives')
    _fields = ('name', 'fields')

    def __init__(self, name, fields, loc=None, directives=None):
        self.loc = loc
        self.name = name
        self.fields = fields
        self.directives = directives

    def __eq__(self, other):
        return self is other or (isinstance(other, InterfaceTypeDefinition) and self.name == other.name and (self.fields == other.fields) and (self.directives == other.directives))

    def __repr__(self):
        return 'InterfaceTypeDefinition(name={self.name!r}, fields={self.fields!r}, directives={self.directives!r})'.format(self=self)

    def __copy__(self):
        return type(self)(self.name, self.fields, self.loc, self.directives)

    def __hash__(self):
        return id(self)