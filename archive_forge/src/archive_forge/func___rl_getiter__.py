import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def __rl_getiter__(self, it):
    return __rl_SafeIter__(it, owner=self.__weakref_ref__(self))