import sys, os
import types
from . import model
from .error import VerificationError
def _load_constant(self, is_int, tp, name, module, check_value=None):
    funcname = '_cffi_const_%s' % name
    if check_value is not None:
        assert is_int
        self._load_known_int_constant(module, funcname)
        value = check_value
    elif is_int:
        BType = self.ffi._typeof_locked('long long*')[0]
        BFunc = self.ffi._typeof_locked('int(*)(long long*)')[0]
        function = module.load_function(BFunc, funcname)
        p = self.ffi.new(BType)
        negative = function(p)
        value = int(p[0])
        if value < 0 and (not negative):
            BLongLong = self.ffi._typeof_locked('long long')[0]
            value += 1 << 8 * self.ffi.sizeof(BLongLong)
    else:
        assert check_value is None
        fntypeextra = '(*)(void)'
        if isinstance(tp, model.StructOrUnion):
            fntypeextra = '*' + fntypeextra
        BFunc = self.ffi._typeof_locked(tp.get_c_name(fntypeextra, name))[0]
        function = module.load_function(BFunc, funcname)
        value = function()
        if isinstance(tp, model.StructOrUnion):
            value = value[0]
    return value