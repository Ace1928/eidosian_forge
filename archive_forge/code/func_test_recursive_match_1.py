import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_recursive_match_1(self):
    """
        Match directives are applied recursively, meaning that they are also
        applied to any content they may have produced themselves:
        """
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <elem py:match="elem">\n            <div class="elem">\n              ${select(\'*\')}\n            </div>\n          </elem>\n          <elem>\n            <subelem>\n              <elem/>\n            </subelem>\n          </elem>\n        </doc>')
    self.assertEqual('<doc>\n          <elem>\n            <div class="elem">\n              <subelem>\n              <elem>\n            <div class="elem">\n            </div>\n          </elem>\n            </subelem>\n            </div>\n          </elem>\n        </doc>', tmpl.generate().render(encoding=None))