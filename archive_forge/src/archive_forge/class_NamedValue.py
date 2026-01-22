import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
class NamedValue(_StrCaching, _StringReferenceCaching, Value):
    """
    The base class for named values.
    """
    name_prefix = '%'
    deduplicate_name = True

    def __init__(self, parent, type, name):
        assert parent is not None
        assert isinstance(type, types.Type)
        self.parent = parent
        self.type = type
        self._set_name(name)

    def _to_string(self):
        buf = []
        if not isinstance(self.type, types.VoidType):
            buf.append('{0} = '.format(self.get_reference()))
        self.descr(buf)
        return ''.join(buf).rstrip()

    def descr(self, buf):
        raise NotImplementedError

    def _get_name(self):
        return self._name

    def _set_name(self, name):
        name = self.parent.scope.register(name, deduplicate=self.deduplicate_name)
        self._name = name
    name = property(_get_name, _set_name)

    def _get_reference(self):
        name = self.name
        if '\\' in name or '"' in name:
            name = name.replace('\\', '\\5c').replace('"', '\\22')
        return '{0}"{1}"'.format(self.name_prefix, name)

    def __repr__(self):
        return "<ir.%s %r of type '%s'>" % (self.__class__.__name__, self.name, self.type)

    @property
    def function_type(self):
        ty = self.type
        if isinstance(ty, types.PointerType):
            ty = self.type.pointee
        if isinstance(ty, types.FunctionType):
            return ty
        else:
            raise TypeError('Not a function: {0}'.format(self.type))