import functools
import unittest
from pecan import expose
from pecan import util
from pecan.compat import getargspec
class deco(object):

    def __init__(self, arg):
        self.arg = arg

    def __call__(self, f):

        @functools.wraps(f)
        def wrapper(*args, **kw):
            assert self.arg == '12345'
            return f(*args, **kw)
        return wrapper