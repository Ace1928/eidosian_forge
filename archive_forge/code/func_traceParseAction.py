import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def traceParseAction(f):
    """Decorator for debugging parse actions."""
    f = _trim_arity(f)

    def z(*paArgs):
        thisFunc = f.func_name
        s, l, t = paArgs[-3:]
        if len(paArgs) > 3:
            thisFunc = paArgs[0].__class__.__name__ + '.' + thisFunc
        sys.stderr.write(">>entering %s(line: '%s', %d, %s)\n" % (thisFunc, line(l, s), l, t))
        try:
            ret = f(*paArgs)
        except Exception as exc:
            sys.stderr.write('<<leaving %s (exception: %s)\n' % (thisFunc, exc))
            raise
        sys.stderr.write('<<leaving %s (ret: %s)\n' % (thisFunc, ret))
        return ret
    try:
        z.__name__ = f.__name__
    except AttributeError:
        pass
    return z