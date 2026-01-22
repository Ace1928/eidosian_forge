import doctest
import unittest
import six
from genshi import HTML
from genshi.builder import Element
from genshi.compat import IS_PYTHON2
from genshi.core import START, END, TEXT, QName, Attrs
from genshi.filters.transform import Transformer, StreamBuffer, ENTER, EXIT, \
import genshi.filters.transform
from genshi.tests.test_utils import doctest_suite
def _apply(self, select, with_attrs=False):
    buffer = StreamBuffer()
    events = buffer.events

    class Trace(object):
        last = None
        trace = []

        def __call__(self, stream):
            for event in stream:
                if events and hash(tuple(events)) != self.last:
                    self.last = hash(tuple(events))
                    self.trace.append(list(events))
                yield event
    trace = Trace()
    output = _transform(FOOBAR, getattr(Transformer(select), self.operation)(buffer).apply(trace), with_attrs=with_attrs)
    simplified = []
    for interval in trace.trace:
        simplified.append(_simplify([(None, e) for e in interval], with_attrs=with_attrs))
    return (output, simplified)