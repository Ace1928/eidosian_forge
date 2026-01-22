import ctypes
import functools
from winappdbg import compat
import sys
def MakeANSIVersion(fn):
    """
    Decorator that generates an ANSI version of a Unicode (wide) only API call.

    @type  fn: callable
    @param fn: Unicode (wide) version of the API function to call.
    """

    @functools.wraps(fn)
    def wrapper(*argv, **argd):
        t_ansi = GuessStringType.t_ansi
        t_unicode = GuessStringType.t_unicode
        v_types = [type(item) for item in argv]
        v_types.extend([type(value) for key, value in compat.iteritems(argd)])
        if t_ansi in v_types:
            argv = list(argv)
            for index in compat.xrange(len(argv)):
                if v_types[index] == t_ansi:
                    argv[index] = t_unicode(argv[index])
            for key, value in argd.items():
                if type(value) == t_ansi:
                    argd[key] = t_unicode(value)
        return fn(*argv, **argd)
    return wrapper