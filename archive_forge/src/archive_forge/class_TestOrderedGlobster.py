import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
class TestOrderedGlobster(TestCase):

    def test_ordered_globs(self):
        """test that the first match in a list is the one found"""
        patterns = ['*.foo', 'bar.*']
        globster = _OrderedGlobster(patterns)
        self.assertEqual('*.foo', globster.match('bar.foo'))
        self.assertEqual(None, globster.match('foo.bar'))
        globster = _OrderedGlobster(reversed(patterns))
        self.assertEqual('bar.*', globster.match('bar.foo'))
        self.assertEqual(None, globster.match('foo.bar'))