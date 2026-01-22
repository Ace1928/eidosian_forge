import io
import time
import unittest
from fastimport import (
from :2
class TestLineBasedParser(unittest.TestCase):

    def test_push_line(self):
        s = io.BytesIO(b'foo\nbar\nbaz\n')
        p = parser.LineBasedParser(s)
        self.assertEqual(b'foo', p.next_line())
        self.assertEqual(b'bar', p.next_line())
        p.push_line(b'bar')
        self.assertEqual(b'bar', p.next_line())
        self.assertEqual(b'baz', p.next_line())
        self.assertEqual(None, p.next_line())

    def test_read_bytes(self):
        s = io.BytesIO(b'foo\nbar\nbaz\n')
        p = parser.LineBasedParser(s)
        self.assertEqual(b'fo', p.read_bytes(2))
        self.assertEqual(b'o\nb', p.read_bytes(3))
        self.assertEqual(b'ar', p.next_line())
        p.push_line(b'bar')
        self.assertEqual(b'baz', p.read_bytes(3))
        self.assertRaises(errors.MissingBytes, p.read_bytes, 10)

    def test_read_until(self):
        return
        s = io.BytesIO(b'foo\nbar\nbaz\nabc\ndef\nghi\n')
        p = parser.LineBasedParser(s)
        self.assertEqual(b'foo\nbar', p.read_until(b'baz'))
        self.assertEqual(b'abc', p.next_line())
        p.push_line(b'abc')
        self.assertEqual(b'def', p.read_until(b'ghi'))
        self.assertRaises(errors.MissingTerminator, p.read_until(b'>>>'))