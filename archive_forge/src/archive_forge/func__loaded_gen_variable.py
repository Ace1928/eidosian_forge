import sys, os
import types
from . import model
from .error import VerificationError
def _loaded_gen_variable(self, tp, name, module, library):
    if isinstance(tp, model.ArrayType):
        if tp.length_is_unknown():
            funcname = '_cffi_sizeof_%s' % (name,)
            BFunc = self.ffi._typeof_locked('size_t(*)(void)')[0]
            function = module.load_function(BFunc, funcname)
            size = function()
            BItemType = self.ffi._get_cached_btype(tp.item)
            length, rest = divmod(size, self.ffi.sizeof(BItemType))
            if rest != 0:
                raise VerificationError('bad size: %r does not seem to be an array of %s' % (name, tp.item))
            tp = tp.resolve_length(length)
        tp_ptr = model.PointerType(tp.item)
        value = self._load_constant(False, tp_ptr, name, module)
        if tp.length is not None:
            BArray = self.ffi._get_cached_btype(tp)
            value = self.ffi.cast(BArray, value)
        setattr(library, name, value)
        type(library)._cffi_dir.append(name)
        return
    funcname = '_cffi_var_%s' % name
    BFunc = self.ffi._typeof_locked(tp.get_c_name('*(*)(void)', name))[0]
    function = module.load_function(BFunc, funcname)
    ptr = function()

    def getter(library):
        return ptr[0]

    def setter(library, value):
        ptr[0] = value
    setattr(type(library), name, property(getter, setter))
    type(library)._cffi_dir.append(name)