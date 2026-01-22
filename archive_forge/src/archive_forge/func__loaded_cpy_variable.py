import sys
from . import model
from .error import VerificationError
from . import _imp_emulation as imp
def _loaded_cpy_variable(self, tp, name, module, library):
    value = getattr(library, name)
    if isinstance(tp, model.ArrayType):
        if tp.length_is_unknown():
            assert isinstance(value, tuple)
            value, size = value
            BItemType = self.ffi._get_cached_btype(tp.item)
            length, rest = divmod(size, self.ffi.sizeof(BItemType))
            if rest != 0:
                raise VerificationError('bad size: %r does not seem to be an array of %s' % (name, tp.item))
            tp = tp.resolve_length(length)
        if tp.length is not None:
            BArray = self.ffi._get_cached_btype(tp)
            value = self.ffi.cast(BArray, value)
            setattr(library, name, value)
        return
    ptr = value
    delattr(library, name)

    def getter(library):
        return ptr[0]

    def setter(library, value):
        ptr[0] = value
    setattr(type(library), name, property(getter, setter))
    type(library)._cffi_dir.append(name)