import ctypes
import functools
from winappdbg import compat
import sys
def MakeWideVersion(fn):
    """
    Decorator that generates a Unicode (wide) version of an ANSI only API call.

    @type  fn: callable
    @param fn: ANSI version of the API function to call.
    """

    @functools.wraps(fn)
    def wrapper(*argv, **argd):
        t_ansi = GuessStringType.t_ansi
        t_unicode = GuessStringType.t_unicode
        v_types = [type(item) for item in argv]
        v_types.extend([type(value) for key, value in compat.iteritems(argd)])
        if t_unicode in v_types:
            argv = list(argv)
            for index in compat.xrange(len(argv)):
                if v_types[index] == t_unicode:
                    argv[index] = t_ansi(argv[index])
            for key, value in argd.items():
                if type(value) == t_unicode:
                    argd[key] = t_ansi(value)
        return fn(*argv, **argd)
    return wrapper