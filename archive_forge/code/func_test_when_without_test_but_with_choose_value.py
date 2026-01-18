import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_when_without_test_but_with_choose_value(self):
    """
        Verify that an `when` directive that doesn't have a `test` attribute
        works as expected as long as the parent `choose` directive has a test
        expression.
        """
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <div py:choose="foo" py:strip="">\n            <py:when>foo</py:when>\n          </div>\n        </doc>')
    self.assertEqual('<doc>\n            foo\n        </doc>', tmpl.generate(foo='Yeah').render(encoding=None))