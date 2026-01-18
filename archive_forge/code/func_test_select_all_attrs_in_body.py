import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_select_all_attrs_in_body(self):
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <div py:match="elem">\n            Hey ${select(\'text()\')} ${select(\'@*\')}\n          </div>\n          <elem title="Cool">Joe</elem>\n        </doc>')
    self.assertEqual('<doc>\n          <div>\n            Hey Joe Cool\n          </div>\n        </doc>', tmpl.generate().render(encoding=None))