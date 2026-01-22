from _sre import MAXREPEAT, MAXGROUPS
class _NamedIntConstant(int):

    def __new__(cls, value, name):
        self = super(_NamedIntConstant, cls).__new__(cls, value)
        self.name = name
        return self

    def __repr__(self):
        return self.name
    __reduce__ = None