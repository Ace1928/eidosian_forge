import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def __rl_apply__(self, func, args, kwds):
    obj = getattr(func, '__self__', None)
    if obj:
        if isinstance(obj, dict) and func.__name__ in ('pop', 'setdefault', 'get', 'popitem'):
            self.__rl_is_allowed_name__(args[0])
    return func(*[a for a in self.__rl_getiter__(args)], **{k: v for k, v in kwds.items()})