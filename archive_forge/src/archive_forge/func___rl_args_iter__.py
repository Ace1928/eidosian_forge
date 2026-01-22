import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def __rl_args_iter__(self, *args):
    if len(args) == 1:
        i = args[0]
        if isinstance(i, __rl_SafeIter__):
            return i
        if not isinstance(i, self.__rl_gen_range__):
            return self.__rl_getiter__(i)
    return self.__rl_getiter__(iter(*args))