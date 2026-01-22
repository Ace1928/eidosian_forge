import os
import sys
def _set_py_limited_api(Extension, kwds):
    """
    Add py_limited_api to kwds if setuptools >= 26 is in use.
    Do not alter the setting if it already exists.
    Setuptools takes care of ignoring the flag on Python 2 and PyPy.

    CPython itself should ignore the flag in a debugging version
    (by not listing .abi3.so in the extensions it supports), but
    it doesn't so far, creating troubles.  That's why we check
    for "not hasattr(sys, 'gettotalrefcount')" (the 2.7 compatible equivalent
    of 'd' not in sys.abiflags). (http://bugs.python.org/issue28401)

    On Windows, with CPython <= 3.4, it's better not to use py_limited_api
    because virtualenv *still* doesn't copy PYTHON3.DLL on these versions.
    Recently (2020) we started shipping only >= 3.5 wheels, though.  So
    we'll give it another try and set py_limited_api on Windows >= 3.5.
    """
    from cffi import recompiler
    if 'py_limited_api' not in kwds and (not hasattr(sys, 'gettotalrefcount')) and recompiler.USE_LIMITED_API:
        import setuptools
        try:
            setuptools_major_version = int(setuptools.__version__.partition('.')[0])
            if setuptools_major_version >= 26:
                kwds['py_limited_api'] = True
        except ValueError:
            kwds['py_limited_api'] = True
    return kwds