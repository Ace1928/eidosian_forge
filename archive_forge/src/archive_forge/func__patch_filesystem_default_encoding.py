import time
import codecs
import sys
def _patch_filesystem_default_encoding(new_enc):
    """Change the Python process global encoding for filesystem names

    The effect is to change how open() and other builtin functions handle
    unicode filenames on posix systems. This should only be done near startup.

    The new encoding string passed to this function must survive until process
    termination, otherwise the interpreter may access uninitialized memory.
    The use of intern() may defer breakage is but is not enough, the string
    object should be secure against module reloading and during teardown.
    """
    try:
        import ctypes
        pythonapi = getattr(ctypes, 'pythonapi', None)
        if pythonapi is not None:
            old_ptr = ctypes.c_void_p.in_dll(pythonapi, 'Py_FileSystemDefaultEncoding')
            has_enc = ctypes.c_int.in_dll(pythonapi, 'Py_HasFileSystemDefaultEncoding')
            as_utf8 = ctypes.PYFUNCTYPE(ctypes.POINTER(ctypes.c_char), ctypes.py_object)(('PyUnicode_AsUTF8', pythonapi))
    except (ImportError, ValueError):
        return
    new_enc = sys.intern(new_enc)
    enc_ptr = as_utf8(new_enc)
    has_enc.value = 1
    old_ptr.value = ctypes.cast(enc_ptr, ctypes.c_void_p).value
    if sys.getfilesystemencoding() != new_enc:
        raise RuntimeError('Failed to change the filesystem default encoding')
    return new_enc