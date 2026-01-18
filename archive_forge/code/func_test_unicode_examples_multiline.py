import ast
import io
import sys
import traceback
import testtools
from testtools.compat import (
from testtools.matchers import (
def test_unicode_examples_multiline(self):
    for u, _, expected in self.unicode_examples:
        actual = text_repr(u, multiline=True)
        self.assertEqual(actual, self.u_prefix + expected)
        self.assertEqual(ast.literal_eval(actual), u)