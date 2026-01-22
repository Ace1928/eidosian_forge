import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
class ArgumentAttributes(AttributeSet):
    _known = MappingProxyType({'byref': True, 'byval': True, 'elementtype': True, 'immarg': False, 'inalloca': True, 'inreg': False, 'nest': False, 'noalias': False, 'nocapture': False, 'nofree': False, 'nonnull': False, 'noundef': False, 'preallocated': True, 'returned': False, 'signext': False, 'sret': True, 'swiftasync': False, 'swifterror': False, 'swiftself': False, 'zeroext': False})

    def __init__(self, args=()):
        self._align = 0
        self._dereferenceable = 0
        self._dereferenceable_or_null = 0
        super(ArgumentAttributes, self).__init__(args)

    def _expand(self, name, typ):
        requires_type = self._known.get(name)
        if requires_type:
            return f'{name}({typ.pointee})'
        else:
            return name

    @property
    def align(self):
        return self._align

    @align.setter
    def align(self, val):
        assert isinstance(val, int) and val >= 0
        self._align = val

    @property
    def dereferenceable(self):
        return self._dereferenceable

    @dereferenceable.setter
    def dereferenceable(self, val):
        assert isinstance(val, int) and val >= 0
        self._dereferenceable = val

    @property
    def dereferenceable_or_null(self):
        return self._dereferenceable_or_null

    @dereferenceable_or_null.setter
    def dereferenceable_or_null(self, val):
        assert isinstance(val, int) and val >= 0
        self._dereferenceable_or_null = val

    def _to_list(self, typ):
        attrs = super()._to_list(typ)
        if self.align:
            attrs.append('align {0:d}'.format(self.align))
        if self.dereferenceable:
            attrs.append('dereferenceable({0:d})'.format(self.dereferenceable))
        if self.dereferenceable_or_null:
            dref = 'dereferenceable_or_null({0:d})'
            attrs.append(dref.format(self.dereferenceable_or_null))
        return attrs