import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_strip_false(self):
    tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <div py:strip="False"><b>foo</b></div>\n        </div>')
    self.assertEqual('<div>\n          <div><b>foo</b></div>\n        </div>', tmpl.generate().render(encoding=None))