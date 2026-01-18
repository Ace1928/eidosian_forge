import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
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