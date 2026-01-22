class VariableDefinition(Node):
    __slots__ = ('loc', 'variable', 'type', 'default_value')
    _fields = ('variable', 'type', 'default_value')

    def __init__(self, variable, type, default_value=None, loc=None):
        self.loc = loc
        self.variable = variable
        self.type = type
        self.default_value = default_value

    def __eq__(self, other):
        return self is other or (isinstance(other, VariableDefinition) and self.variable == other.variable and (self.type == other.type) and (self.default_value == other.default_value))

    def __repr__(self):
        return 'VariableDefinition(variable={self.variable!r}, type={self.type!r}, default_value={self.default_value!r})'.format(self=self)

    def __copy__(self):
        return type(self)(self.variable, self.type, self.default_value, self.loc)

    def __hash__(self):
        return id(self)