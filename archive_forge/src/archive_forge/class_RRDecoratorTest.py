import unittest
import cachetools.func
class RRDecoratorTest(unittest.TestCase, DecoratorTestMixin):
    DECORATOR = staticmethod(cachetools.func.rr_cache)