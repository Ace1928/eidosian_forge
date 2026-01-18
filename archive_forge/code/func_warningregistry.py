import functools
import sys
import types
import warnings
import unittest
def warningregistry(func):

    def wrapper(*args, **kws):
        missing = []
        saved = getattr(warnings, '__warningregistry__', missing).copy()
        try:
            return func(*args, **kws)
        finally:
            if saved is missing:
                try:
                    del warnings.__warningregistry__
                except AttributeError:
                    pass
            else:
                warnings.__warningregistry__ = saved
    return wrapper