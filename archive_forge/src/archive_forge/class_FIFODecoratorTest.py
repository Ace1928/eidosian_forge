import unittest
import cachetools.func
class FIFODecoratorTest(unittest.TestCase, DecoratorTestMixin):
    DECORATOR = staticmethod(cachetools.func.fifo_cache)