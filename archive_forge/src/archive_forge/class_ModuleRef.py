from ctypes import (c_char_p, byref, POINTER, c_bool, create_string_buffer,
from llvmlite.binding import ffi
from llvmlite.binding.linker import link_modules
from llvmlite.binding.common import _decode_string, _encode_string
from llvmlite.binding.value import ValueRef, TypeRef
from llvmlite.binding.context import get_global_context
class ModuleRef(ffi.ObjectRef):
    """
    A reference to a LLVM module.
    """

    def __init__(self, module_ptr, context):
        super(ModuleRef, self).__init__(module_ptr)
        self._context = context

    def __str__(self):
        with ffi.OutputString() as outstr:
            ffi.lib.LLVMPY_PrintModuleToString(self, outstr)
            return str(outstr)

    def as_bitcode(self):
        """
        Return the module's LLVM bitcode, as a bytes object.
        """
        ptr = c_char_p(None)
        size = c_size_t(-1)
        ffi.lib.LLVMPY_WriteBitcodeToString(self, byref(ptr), byref(size))
        if not ptr:
            raise MemoryError
        try:
            assert size.value >= 0
            return string_at(ptr, size.value)
        finally:
            ffi.lib.LLVMPY_DisposeString(ptr)

    def _dispose(self):
        self._capi.LLVMPY_DisposeModule(self)

    def get_function(self, name):
        """
        Get a ValueRef pointing to the function named *name*.
        NameError is raised if the symbol isn't found.
        """
        p = ffi.lib.LLVMPY_GetNamedFunction(self, _encode_string(name))
        if not p:
            raise NameError(name)
        return ValueRef(p, 'function', dict(module=self))

    def get_global_variable(self, name):
        """
        Get a ValueRef pointing to the global variable named *name*.
        NameError is raised if the symbol isn't found.
        """
        p = ffi.lib.LLVMPY_GetNamedGlobalVariable(self, _encode_string(name))
        if not p:
            raise NameError(name)
        return ValueRef(p, 'global', dict(module=self))

    def get_struct_type(self, name):
        """
        Get a TypeRef pointing to a structure type named *name*.
        NameError is raised if the struct type isn't found.
        """
        p = ffi.lib.LLVMPY_GetNamedStructType(self, _encode_string(name))
        if not p:
            raise NameError(name)
        return TypeRef(p)

    def verify(self):
        """
        Verify the module IR's correctness.  RuntimeError is raised on error.
        """
        with ffi.OutputString() as outmsg:
            if ffi.lib.LLVMPY_VerifyModule(self, outmsg):
                raise RuntimeError(str(outmsg))

    @property
    def name(self):
        """
        The module's identifier.
        """
        return _decode_string(ffi.lib.LLVMPY_GetModuleName(self))

    @name.setter
    def name(self, value):
        ffi.lib.LLVMPY_SetModuleName(self, _encode_string(value))

    @property
    def source_file(self):
        """
        The module's original source file name
        """
        return _decode_string(ffi.lib.LLVMPY_GetModuleSourceFileName(self))

    @property
    def data_layout(self):
        """
        This module's data layout specification, as a string.
        """
        with ffi.OutputString(owned=False) as outmsg:
            ffi.lib.LLVMPY_GetDataLayout(self, outmsg)
            return str(outmsg)

    @data_layout.setter
    def data_layout(self, strrep):
        ffi.lib.LLVMPY_SetDataLayout(self, create_string_buffer(strrep.encode('utf8')))

    @property
    def triple(self):
        """
        This module's target "triple" specification, as a string.
        """
        with ffi.OutputString(owned=False) as outmsg:
            ffi.lib.LLVMPY_GetTarget(self, outmsg)
            return str(outmsg)

    @triple.setter
    def triple(self, strrep):
        ffi.lib.LLVMPY_SetTarget(self, create_string_buffer(strrep.encode('utf8')))

    def link_in(self, other, preserve=False):
        """
        Link the *other* module into this one.  The *other* module will
        be destroyed unless *preserve* is true.
        """
        if preserve:
            other = other.clone()
        link_modules(self, other)

    @property
    def global_variables(self):
        """
        Return an iterator over this module's global variables.
        The iterator will yield a ValueRef for each global variable.

        Note that global variables don't include functions
        (a function is a "global value" but not a "global variable" in
         LLVM parlance)
        """
        it = ffi.lib.LLVMPY_ModuleGlobalsIter(self)
        return _GlobalsIterator(it, dict(module=self))

    @property
    def functions(self):
        """
        Return an iterator over this module's functions.
        The iterator will yield a ValueRef for each function.
        """
        it = ffi.lib.LLVMPY_ModuleFunctionsIter(self)
        return _FunctionsIterator(it, dict(module=self))

    @property
    def struct_types(self):
        """
        Return an iterator over the struct types defined in
        the module. The iterator will yield a TypeRef.
        """
        it = ffi.lib.LLVMPY_ModuleTypesIter(self)
        return _TypesIterator(it, dict(module=self))

    def clone(self):
        return ModuleRef(ffi.lib.LLVMPY_CloneModule(self), self._context)