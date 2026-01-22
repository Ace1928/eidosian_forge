import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def __rl_hasattr__(self, obj, name):
    try:
        self.__rl_getattr__(obj, name)
    except (AttributeError, BadCode, TypeError):
        return False
    return True