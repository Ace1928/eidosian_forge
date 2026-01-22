import os
from .dependencies import ctypes
def _load_dll(name, timeout=10):
    """Load a DLL with a timeout

    On some platforms and some DLLs (notably Windows GitHub Actions with
    Python 3.5, 3.6, and 3.7 and the msvcr90.dll) we have observed
    behavior where the ctypes.CDLL() call hangs indefinitely. This uses
    multiprocessing to attempt the import in a subprocess (with a
    timeout) and then only calls the import in the main process if the
    subprocess succeeded.

    Performance note: CtypesEnviron only ever attempts to load a DLL
    once (the DLL reference is then held in a class attribute), and this
    interface only spawns the subprocess if ctypes.util.find_library
    actually locates the target library. This will have a measurable
    impact on Windows (where the DLLs exist), but not on other platforms.

    The default timeout of 10 is arbitrary. For simple situations, 1
    seems adequate. However, more complex examples have been observed
    that needed timeout==5. Using a default of 10 is simply doubling
    that observed case.

    """
    if not ctypes.util.find_library(name):
        return (False, None)
    import multiprocessing
    if _load_dll.pool is None:
        try:
            _load_dll.pool = multiprocessing.Pool(1)
        except AssertionError:
            import multiprocessing.dummy
            _load_dll.pool = multiprocessing.dummy.Pool(1)
    job = _load_dll.pool.apply_async(_attempt_ctypes_cdll, (name,))
    try:
        result = job.get(timeout)
    except multiprocessing.TimeoutError:
        result = False
        _load_dll.pool.terminate()
        _load_dll.pool = None
    if result:
        return (result, ctypes.CDLL(name))
    else:
        return (result, None)