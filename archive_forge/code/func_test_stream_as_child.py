import doctest
import unittest
from genshi.builder import Element, tag
from genshi.core import Attrs, Markup, Stream
from genshi.input import XML
def test_stream_as_child(self):
    events = list(tag.span(XML('<b>Foo</b>')).generate())
    self.assertEqual(5, len(events))
    self.assertEqual((Stream.START, ('span', ())), events[0][:2])
    self.assertEqual((Stream.START, ('b', ())), events[1][:2])
    self.assertEqual((Stream.TEXT, 'Foo'), events[2][:2])
    self.assertEqual((Stream.END, 'b'), events[3][:2])
    self.assertEqual((Stream.END, 'span'), events[4][:2])