import pickle
from collections import namedtuple
class RecordLevel:
    __slots__ = ('name', 'no', 'icon')

    def __init__(self, name, no, icon):
        self.name = name
        self.no = no
        self.icon = icon

    def __repr__(self):
        return '(name=%r, no=%r, icon=%r)' % (self.name, self.no, self.icon)

    def __format__(self, spec):
        return self.name.__format__(spec)