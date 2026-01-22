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
def _filter(self, select, html=FOOBAR):
    """Returns a list of lists of filtered elements."""
    output = []

    def filtered(stream):
        interval = []
        output.append(interval)
        for event in stream:
            interval.append(event)
            yield event
    _transform(html, Transformer(select).filter(filtered))
    simplified = []
    for sub in output:
        simplified.append(_simplify([(None, event) for event in sub]))
    return simplified