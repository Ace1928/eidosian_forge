import unittest
import cachetools.func
class TTLDecoratorTest(unittest.TestCase, DecoratorTestMixin):
    DECORATOR = staticmethod(cachetools.func.ttl_cache)