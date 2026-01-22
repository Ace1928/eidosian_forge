import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
class AtomicType(object):

    def __init__(self, name, default, python_types):
        self.name = name
        self.default = default
        self.python_types = python_types

    @staticmethod
    def from_string(s):
        if s != 'void':
            for atomic_type in ATOMIC_TYPES:
                if s == atomic_type.name:
                    return atomic_type
        raise error.Error('"%s" is not an atomic-type' % s, s)

    @staticmethod
    def from_json(json):
        if not isinstance(json, str):
            raise error.Error('atomic-type expected', json)
        else:
            return AtomicType.from_string(json)

    def __str__(self):
        return self.name

    def to_string(self):
        return self.name

    def to_rvalue_string(self):
        if self == StringType:
            return 's->' + self.name
        return self.name

    def to_lvalue_string(self):
        if self == StringType:
            return 's'
        return self.name

    def to_json(self):
        return self.name

    def default_atom(self):
        return ovs.db.data.Atom(self, self.default)