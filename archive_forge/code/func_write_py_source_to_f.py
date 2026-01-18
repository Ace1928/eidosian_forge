import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def write_py_source_to_f(self, f):
    self._f = f
    prnt = self._prnt
    prnt('# auto-generated file')
    prnt('import _cffi_backend')
    num_includes = len(self.ffi._included_ffis or ())
    for i in range(num_includes):
        ffi_to_include = self.ffi._included_ffis[i]
        try:
            included_module_name, included_source = ffi_to_include._assigned_source[:2]
        except AttributeError:
            raise VerificationError('ffi object %r includes %r, but the latter has not been prepared with set_source()' % (self.ffi, ffi_to_include))
        if included_source is not None:
            raise VerificationError('not implemented yet: ffi.include() of a C-based ffi inside a Python-based ffi')
        prnt('from %s import ffi as _ffi%d' % (included_module_name, i))
    prnt()
    prnt("ffi = _cffi_backend.FFI('%s'," % (self.module_name,))
    prnt('    _version = 0x%x,' % (self._version,))
    self._version = None
    self.cffi_types = tuple(self.cffi_types)
    types_lst = [op.as_python_bytes() for op in self.cffi_types]
    prnt('    _types = %s,' % (self._to_py(''.join(types_lst)),))
    typeindex2type = dict([(i, tp) for tp, i in self._typesdict.items()])
    for step_name in self.ALL_STEPS:
        lst = self._lsts[step_name]
        if len(lst) > 0 and step_name != 'field':
            prnt('    _%ss = %s,' % (step_name, self._to_py(lst)))
    if num_includes > 0:
        prnt('    _includes = (%s,),' % (', '.join(['_ffi%d' % i for i in range(num_includes)]),))
    prnt(')')