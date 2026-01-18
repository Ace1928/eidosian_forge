import py
import sys
from inspect import CO_VARARGS, CO_VARKEYWORDS, isclass
import traceback
def unpatch_builtins(assertion=True, compile=True):
    """ remove compile and AssertionError builtins from Python builtins. """
    if assertion:
        py.builtin.builtins.AssertionError = oldbuiltins['AssertionError'].pop()
    if compile:
        py.builtin.builtins.compile = oldbuiltins['compile'].pop()