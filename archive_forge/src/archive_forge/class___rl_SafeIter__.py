import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
class __rl_SafeIter__:

    def __init__(self, it, owner):
        self.__rl_iter__ = owner().__rl_real_iter__(it)
        self.__rl_owner__ = owner

    def __iter__(self):
        return self

    def __next__(self):
        self.__rl_owner__().__rl_check__()
        return next(self.__rl_iter__)
    next = __next__