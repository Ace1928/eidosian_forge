import ctypes
from ctypes import POINTER, c_bool, c_char_p, c_uint8, c_uint64, c_size_t
from llvmlite.binding import ffi, targets
class JITLibraryBuilder:
    """
    Create a library for linking by OrcJIT

    OrcJIT operates like a linker: a number of compilation units and
    dependencies are collected together and linked into a single dynamic library
    that can export functions to other libraries or to be consumed directly as
    entry points into JITted code. The native OrcJIT has a lot of memory
    management complications so this API is designed to work well with Python's
    garbage collection.

    The creation of a new library is a bit like a linker command line where
    compilation units, mostly as LLVM IR, and previously constructed libraries
    are linked together, then loaded into memory, and the addresses of exported
    symbols are extracted. Any static initializers are run and the exported
    addresses and a resource tracker is produced. As long as the resource
    tracker is referenced somewhere in Python, the exported addresses will be
    valid. Once the resource tracker is garbage collected, the static
    destructors will run and library will be unloaded from memory.
    """

    def __init__(self):
        self.__entries = []
        self.__exports = set()
        self.__imports = {}

    def add_ir(self, llvmir):
        """
        Adds a compilation unit to the library using LLVM IR as the input
        format.

        This takes a string or an object that can be converted to a string,
        including IRBuilder, that contains LLVM IR.
        """
        self.__entries.append((0, str(llvmir).encode('utf-8')))
        return self

    def add_native_assembly(self, asm):
        """
        Adds a compilation unit to the library using native assembly as the
        input format.

        This takes a string or an object that can be converted to a string that
        contains native assembly, which will be
        parsed by LLVM.
        """
        self.__entries.append((1, str(asm).encode('utf-8')))
        return self

    def add_object_img(self, data):
        """
        Adds a compilation unit to the library using pre-compiled object code.

        This takes the bytes of the contents of an object artifact which will be
        loaded by LLVM.
        """
        self.__entries.append((2, bytes(data)))
        return self

    def add_object_file(self, file_path):
        """
        Adds a compilation unit to the library using pre-compiled object file.

        This takes a string or path-like object that references an object file
        which will be loaded by LLVM.
        """
        with open(file_path, 'rb') as f:
            self.__entries.append((2, f.read()))
        return self

    def add_jit_library(self, name):
        """
        Adds an existing JIT library as prerequisite.

        The name of the library must match the one provided in a previous link
        command.
        """
        self.__entries.append((3, str(name).encode('utf-8')))
        return self

    def add_current_process(self):
        """
        Allows the JITted library to access symbols in the current binary.

        That is, it allows exporting the current binary's symbols, including
        loaded libraries, as imports to the JITted
        library.
        """
        self.__entries.append((3, b''))
        return self

    def import_symbol(self, name, address):
        """
        Register the *address* of global symbol *name*.  This will make
        it usable (e.g. callable) from LLVM-compiled functions.
        """
        self.__imports[str(name)] = c_uint64(address)
        return self

    def export_symbol(self, name):
        """
        During linking, extract the address of a symbol that was defined in one
        of the compilation units.

        This allows getting symbols, functions or global variables, out of the
        JIT linked library. The addresses will be
        available when the link method is called.
        """
        self.__exports.add(str(name))
        return self

    def link(self, lljit, library_name):
        """
        Link all the current compilation units into a JITted library and extract
        the address of exported symbols.

        An instance of the OrcJIT instance must be provided and this will be the
        scope that is used to find other JITted libraries that are dependencies
        and also be the place where this library will be defined.

        After linking, the method will return a resource tracker that keeps the
        library alive. This tracker also knows the addresses of any exported
        symbols that were requested.

        The addresses will be valid as long as the resource tracker is
        referenced.

        When the resource tracker is destroyed, the library will be cleaned up,
        however, the name of the library cannot be reused.
        """
        assert not lljit.closed, 'Cannot add to closed JIT'
        encoded_library_name = str(library_name).encode('utf-8')
        assert len(encoded_library_name) > 0, 'Library cannot be empty'
        elements = (_LinkElement * len(self.__entries))()
        for idx, (kind, value) in enumerate(self.__entries):
            elements[idx].element_kind = c_uint8(kind)
            elements[idx].value = c_char_p(value)
            elements[idx].value_len = c_size_t(len(value))
        exports = (_SymbolAddress * len(self.__exports))()
        for idx, name in enumerate(self.__exports):
            exports[idx].name = name.encode('utf-8')
        imports = (_SymbolAddress * len(self.__imports))()
        for idx, (name, addr) in enumerate(self.__imports.items()):
            imports[idx].name = name.encode('utf-8')
            imports[idx].address = addr
        with ffi.OutputString() as outerr:
            tracker = lljit._capi.LLVMPY_LLJIT_Link(lljit._ptr, encoded_library_name, elements, len(self.__entries), imports, len(self.__imports), exports, len(self.__exports), outerr)
            if not tracker:
                raise RuntimeError(str(outerr))
        return ResourceTracker(tracker, library_name, {name: exports[idx].address for idx, name in enumerate(self.__exports)})