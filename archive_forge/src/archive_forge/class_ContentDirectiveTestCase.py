import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
class ContentDirectiveTestCase(unittest.TestCase):
    """Tests for the `py:content` template directive."""

    def test_as_element(self):
        try:
            MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n              <py:content foo="">Foo</py:content>\n            </doc>', filename='test.html').generate()
            self.fail('Expected TemplateSyntaxError')
        except TemplateSyntaxError as e:
            self.assertEqual('test.html', e.filename)
            self.assertEqual(2, e.lineno)