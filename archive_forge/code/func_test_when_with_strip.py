import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_when_with_strip(self):
    """
        Verify that a when directive with a strip directive actually strips of
        the outer element.
        """
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <div py:choose="" py:strip="">\n            <span py:otherwise="">foo</span>\n          </div>\n        </doc>')
    self.assertEqual('<doc>\n            <span>foo</span>\n        </doc>', tmpl.generate().render(encoding=None))