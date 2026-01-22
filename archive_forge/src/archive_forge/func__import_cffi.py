from . import _ccallback_c
import ctypes
def _import_cffi():
    global ffi, CData
    if ffi is not None:
        return
    try:
        import cffi
        ffi = cffi.FFI()
        CData = ffi.CData
    except ImportError:
        ffi = False