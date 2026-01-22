import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
class __rl_dict__(dict):

    def __new__(cls, *args, **kwds):
        if len(args) == 1 and (not isinstance(args[0], dict)):
            try:
                it = self.__real_iter__(args[0])
            except TypeError:
                pass
            else:
                args = (self.__rl_getiter__(it),)
        return dict.__new__(cls, *args, **kwds)