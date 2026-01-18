import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_recursive_match_3(self):
    tmpl = MarkupTemplate('<test xmlns:py="http://genshi.edgewall.org/">\n          <py:match path="b[@type=\'bullet\']">\n            <bullet>${select(\'*|text()\')}</bullet>\n          </py:match>\n          <py:match path="group[@type=\'bullet\']">\n            <ul>${select(\'*\')}</ul>\n          </py:match>\n          <py:match path="b">\n            <generic>${select(\'*|text()\')}</generic>\n          </py:match>\n\n          <b>\n            <group type="bullet">\n              <b type="bullet">1</b>\n              <b type="bullet">2</b>\n            </group>\n          </b>\n        </test>\n        ')
    self.assertEqual('<test>\n            <generic>\n            <ul><bullet>1</bullet><bullet>2</bullet></ul>\n          </generic>\n        </test>', tmpl.generate().render(encoding=None))