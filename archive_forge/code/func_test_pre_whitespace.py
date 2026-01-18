import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
def test_pre_whitespace(self):
    content = '\nHey <em>there</em>.  \n\n    I am indented.\n'
    stream = XML('<pre>%s</pre>' % content)
    output = stream.render(HTMLSerializer, encoding=None)
    self.assertEqual('<pre>%s</pre>' % content, output)