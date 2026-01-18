import gc
from twisted.internet import defer
from twisted.python.compat import networkString
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase, TestCase
from twisted.web import resource, util
from twisted.web.error import FlattenerError
from twisted.web.http import FOUND
from twisted.web.server import Request
from twisted.web.template import TagLoader, flattenString, tags
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
from twisted.web.util import (
def test_sourceFragmentElement(self):
    """
        L{_SourceFragmentElement} renders source lines at and around the line
        number indicated by a frame object.
        """
    element = _SourceFragmentElement(TagLoader(tags.div(tags.span(render='lineNumber'), tags.span(render='sourceLine'), render='sourceLines')), self.frame)
    source = [' \xa0 \xa0message = "This is a problem"', ' \xa0 \xa0raise Exception(message)', '']
    d = flattenString(None, element)
    stringToCheckFor = ''
    for lineNumber, sourceLine in enumerate(source):
        template = '<div class="snippet{}Line"><span>{}</span><span>{}</span></div>'
        if lineNumber <= 1:
            stringToCheckFor += template.format(['', 'Highlight'][lineNumber == 1], self.base + lineNumber, ' \xa0' * 4 + sourceLine)
        else:
            stringToCheckFor += template.format('', self.base + lineNumber, '' + sourceLine)
    bytesToCheckFor = stringToCheckFor.encode('utf8')
    d.addCallback(self.assertEqual, bytesToCheckFor)
    return d