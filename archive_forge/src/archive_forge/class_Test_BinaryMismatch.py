import re
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._basic import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class Test_BinaryMismatch(TestCase):
    """Mismatches from binary comparisons need useful describe output"""
    _long_string = 'This is a longish multiline non-ascii string\n§'
    _long_b = _b(_long_string)
    _long_u = _long_string

    class CustomRepr:

        def __init__(self, repr_string):
            self._repr_string = repr_string

        def __repr__(self):
            return '<object ' + self._repr_string + '>'

    def test_short_objects(self):
        o1, o2 = (self.CustomRepr('a'), self.CustomRepr('b'))
        mismatch = _BinaryMismatch(o1, '!~', o2)
        self.assertEqual(mismatch.describe(), f'{o1!r} !~ {o2!r}')

    def test_short_mixed_strings(self):
        b, u = (_b('§'), '§')
        mismatch = _BinaryMismatch(b, '!~', u)
        self.assertEqual(mismatch.describe(), f'{b!r} !~ {u!r}')

    def test_long_bytes(self):
        one_line_b = self._long_b.replace(_b('\n'), _b(' '))
        mismatch = _BinaryMismatch(one_line_b, '!~', self._long_b)
        self.assertEqual(mismatch.describe(), '{}:\nreference = {}\nactual    = {}\n'.format('!~', text_repr(self._long_b, multiline=True), text_repr(one_line_b)))

    def test_long_unicode(self):
        one_line_u = self._long_u.replace('\n', ' ')
        mismatch = _BinaryMismatch(one_line_u, '!~', self._long_u)
        self.assertEqual(mismatch.describe(), '{}:\nreference = {}\nactual    = {}\n'.format('!~', text_repr(self._long_u, multiline=True), text_repr(one_line_u)))

    def test_long_mixed_strings(self):
        mismatch = _BinaryMismatch(self._long_b, '!~', self._long_u)
        self.assertEqual(mismatch.describe(), '{}:\nreference = {}\nactual    = {}\n'.format('!~', text_repr(self._long_u, multiline=True), text_repr(self._long_b, multiline=True)))

    def test_long_bytes_and_object(self):
        obj = object()
        mismatch = _BinaryMismatch(self._long_b, '!~', obj)
        self.assertEqual(mismatch.describe(), '{}:\nreference = {}\nactual    = {}\n'.format('!~', repr(obj), text_repr(self._long_b, multiline=True)))

    def test_long_unicode_and_object(self):
        obj = object()
        mismatch = _BinaryMismatch(self._long_u, '!~', obj)
        self.assertEqual(mismatch.describe(), '{}:\nreference = {}\nactual    = {}\n'.format('!~', repr(obj), text_repr(self._long_u, multiline=True)))