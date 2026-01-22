import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def __rl_getattr__(self, obj, a, *args):
    if isinstance(obj, strTypes) and a == 'format':
        raise BadCode('%s.format is not implemented' % type(obj))
    self.__rl_is_allowed_name__(a)
    return getattr(obj, a, *args)