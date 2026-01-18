import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_otherwise(self):
    tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/" py:choose="">\n          <span py:when="False">hidden</span>\n          <span py:otherwise="">hello</span>\n        </div>')
    self.assertEqual('<div>\n          <span>hello</span>\n        </div>', tmpl.generate().render(encoding=None))