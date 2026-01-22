import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def __rl_getitem__(self, obj, a):
    if type(a) is self.__slicetype__:
        if a.step is not None:
            v = obj[a]
        else:
            start = a.start
            stop = a.stop
            if start is None:
                start = 0
            if stop is None:
                v = obj[start:]
            else:
                v = obj[start:stop]
        return v
    elif isinstance(a, strTypes):
        self.__rl_is_allowed_name__(a)
        return obj[a]
    return obj[a]