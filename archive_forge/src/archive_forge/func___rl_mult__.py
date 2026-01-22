import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def __rl_mult__(self, a, b):
    if hasattr(a, '__len__') and b * len(a) > self.__rl_max_len__ or (hasattr(b, '__len__') and a * len(b) > self.__rl_max_len__):
        raise BadCode('excessive length')
    return a * b