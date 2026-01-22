class FieldDefinition(Node):
    __slots__ = ('loc', 'name', 'arguments', 'type', 'directives')
    _fields = ('name', 'arguments', 'type')

    def __init__(self, name, arguments, type, loc=None, directives=None):
        self.loc = loc
        self.name = name
        self.arguments = arguments
        self.type = type
        self.directives = directives

    def __eq__(self, other):
        return self is other or (isinstance(other, FieldDefinition) and self.name == other.name and (self.arguments == other.arguments) and (self.type == other.type) and (self.directives == other.directives))

    def __repr__(self):
        return 'FieldDefinition(name={self.name!r}, arguments={self.arguments!r}, type={self.type!r})'.format(self=self)

    def __copy__(self):
        return type(self)(self.name, self.arguments, self.type, self.loc, self.directives)

    def __hash__(self):
        return id(self)