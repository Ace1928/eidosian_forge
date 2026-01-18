import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_with_empty_value(self):
    """
        Verify that an empty py:with works (useless, but legal)
        """
    tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <span py:with="">Text</span></div>')
    self.assertEqual('<div>\n          <span>Text</span></div>', tmpl.generate().render(encoding=None))