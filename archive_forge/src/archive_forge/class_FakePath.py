import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
class FakePath(Path):

    def __init__(self, strategy):
        self.strategy = strategy

    def test(self, ignore_context=False):
        return self.strategy.test(ignore_context)