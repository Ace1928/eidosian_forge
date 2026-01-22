from __future__ import absolute_import
import math, sys
class CythonCImports(object):
    """
    Simplistic module mock to make cimports sort-of work in Python code.
    """

    def __init__(self, module):
        self.__path__ = []
        self.__file__ = None
        self.__name__ = module
        self.__package__ = module

    def __getattr__(self, item):
        if item.startswith('__') and item.endswith('__'):
            raise AttributeError(item)
        try:
            return __import__(item)
        except ImportError:
            import sys
            ex = AttributeError(item)
            if sys.version_info >= (3, 0):
                ex.__cause__ = None
            raise ex