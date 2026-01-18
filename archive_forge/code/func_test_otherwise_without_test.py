import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_otherwise_without_test(self):
    """
        Verify that an `otherwise` directive can be used without a `test`
        attribute.
        """
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <div py:choose="" py:strip="">\n            <py:otherwise>foo</py:otherwise>\n          </div>\n        </doc>')
    self.assertEqual('<doc>\n            foo\n        </doc>', tmpl.generate().render(encoding=None))