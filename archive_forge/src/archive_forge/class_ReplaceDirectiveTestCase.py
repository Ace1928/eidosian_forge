import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
class ReplaceDirectiveTestCase(unittest.TestCase):
    """Tests for the `py:replace` template directive."""

    def test_replace_with_empty_value(self):
        """
        Verify that the directive raises an apprioriate exception when an empty
        expression is supplied.
        """
        try:
            MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n              <elem py:replace="">Foo</elem>\n            </doc>', filename='test.html').generate()
            self.fail('Expected TemplateSyntaxError')
        except TemplateSyntaxError as e:
            self.assertEqual('test.html', e.filename)
            self.assertEqual(2, e.lineno)

    def test_as_element(self):
        tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <py:replace value="title" />\n        </div>', filename='test.html')
        self.assertEqual('<div>\n          Test\n        </div>', tmpl.generate(title='Test').render(encoding=None))