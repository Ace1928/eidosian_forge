import sys
import os
import ctypes
import optparse
from winappdbg import win32
from winappdbg import compat
class Regenerator(object):
    """
    Calls a generator and iterates it. When it's finished iterating, the
    generator is called again. This allows you to iterate a generator more
    than once (well, sort of).
    """

    def __init__(self, g_function, *v_args, **d_args):
        """
        @type  g_function: function
        @param g_function: Function that when called returns a generator.

        @type  v_args: tuple
        @param v_args: Variable arguments to pass to the generator function.

        @type  d_args: dict
        @param d_args: Variable arguments to pass to the generator function.
        """
        self.__g_function = g_function
        self.__v_args = v_args
        self.__d_args = d_args
        self.__g_object = None

    def __iter__(self):
        """x.__iter__() <==> iter(x)"""
        return self

    def next(self):
        """x.next() -> the next value, or raise StopIteration"""
        if self.__g_object is None:
            self.__g_object = self.__g_function(*self.__v_args, **self.__d_args)
        try:
            return self.__g_object.next()
        except StopIteration:
            self.__g_object = None
            raise