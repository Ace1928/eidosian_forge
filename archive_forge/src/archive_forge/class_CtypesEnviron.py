import os
from .dependencies import ctypes
class CtypesEnviron(object):
    """A context manager for managing environment variables

    This class provides a simplified interface for consistently setting
    and restoring environment variables, with special handling to ensure
    consistency with the C Runtime Library environment on Windows
    platforms.

    `os.environ` reflects the current python environment variables, and
    will be passed to subprocesses.  However, it does not reflect the C
    Runtime Library (MSVCRT) environment on Windows platforms.  This can
    be problemmatic as DLLs loaded through the CTYPES interface will see
    the MSVCRT environment and not os.environ.  This class provides a
    way to manage environment variables and pass changes to both
    os.environ and the MSVCRT runtime.

    This class implements a context manager API, so that clients can
    temporarily change - and then subsequently restore - the
    environment.

    .. testcode::
       :hide:

       import os
       from pyomo.common.env import TemporaryEnv
       orig_env_val = os.environ.get('TEMP_ENV_VAR', None)

    .. doctest::

       >>> os.environ['TEMP_ENV_VAR'] = 'original value'
       >>> print(os.environ['TEMP_ENV_VAR'])
       original value

       >>> with CtypesEnviron(TEMP_ENV_VAR='temporary value'):
       ...    print(os.envion['TEMP_ENV_VAR'])
       temporary value

       >>> print(os.environ['TEMP_ENV_VAR'])
       original value

    .. testcode::
       :hide:

       if orig_env_val is None:
           del os.environ['TEMP_ENV_VAR']
       else:
           os.environ['TEMP_ENV_VAR'] = orig_env_val

    """
    DLLs = [_Win32DLL('kernel32'), _MsvcrtDLL(getattr(ctypes.util, 'find_msvcrt', lambda: None)()), _MsvcrtDLL('api-ms-win-crt-environment-l1-1-0'), _MsvcrtDLL('msvcrt'), _MsvcrtDLL('msvcr120'), _MsvcrtDLL('msvcr110'), _MsvcrtDLL('msvcr100'), _MsvcrtDLL('msvcr90'), _MsvcrtDLL('msvcr80'), _MsvcrtDLL('msvcr71'), _MsvcrtDLL('msvcr70')]

    def __init__(self, **kwds):
        self.interfaces = [_RestorableEnvironInterface(_OSEnviron())]
        self.interfaces.extend((_RestorableEnvironInterface(dll) for dll in self.DLLs if dll.available()))
        if _load_dll.pool is not None:
            _load_dll.pool.terminate()
            _load_dll.pool = None
        for k, v in kwds.items():
            self[k] = v

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.restore()

    def restore(self):
        """Restore the environment to the original state

        This restores all environment variables modified through this
        object to the state they were in before this instance made any
        changes.  Note that any changes made directly to os.environ
        outside this instance will not be detected / undone.

        """
        for lib in reversed(self.interfaces):
            lib.restore()

    def __getitem__(self, key):
        """Return the current environment variable value from os.environ"""
        return os.environ[key]

    def __contains__(self, key):
        """Return True if the key is in os.environ"""
        return key in os.environ

    def __setitem__(self, key, val):
        """Set an environment variable in all known runtime environments"""
        for lib in self.interfaces:
            lib[key] = val

    def __delitem__(self, key):
        """Remove an environment variable from all known runtime environments"""
        for lib in self.interfaces:
            del lib[key]