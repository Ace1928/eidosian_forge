import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
class FunctionPtrType(BaseFunctionType):
    _base_pattern = '(*&)(%s)'

    def build_backend_type(self, ffi, finishlist):
        result = self.result.get_cached_btype(ffi, finishlist)
        args = []
        for tp in self.args:
            args.append(tp.get_cached_btype(ffi, finishlist))
        abi_args = ()
        if self.abi == '__stdcall':
            if not self.ellipsis:
                try:
                    abi_args = (ffi._backend.FFI_STDCALL,)
                except AttributeError:
                    pass
        return global_cache(self, ffi, 'new_function_type', tuple(args), result, self.ellipsis, *abi_args)

    def as_raw_function(self):
        return RawFunctionType(self.args, self.result, self.ellipsis, self.abi)