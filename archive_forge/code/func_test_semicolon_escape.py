import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_semicolon_escape(self):
    tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <py:with vars="x = \'here is a semicolon: ;\'; y = \'here are two semicolons: ;;\' ;">\n            ${x}\n            ${y}\n          </py:with>\n        </div>')
    self.assertEqual('<div>\n            here is a semicolon: ;\n            here are two semicolons: ;;\n        </div>', tmpl.generate().render(encoding=None))