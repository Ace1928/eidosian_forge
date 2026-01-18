import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_when_outside_choose(self):
    """
        Verify that a `when` directive outside of a `choose` directive is
        reported as an error.
        """
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <div py:when="xy" />\n        </doc>')
    self.assertRaises(TemplateRuntimeError, str, tmpl.generate())