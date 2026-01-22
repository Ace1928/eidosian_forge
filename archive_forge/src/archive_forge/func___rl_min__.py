import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def __rl_min__(self, arg, *args, **kwds):
    if args:
        arg = [arg]
        arg.extend(args)
    return min(self.__rl_args_iter__(arg), **kwds)