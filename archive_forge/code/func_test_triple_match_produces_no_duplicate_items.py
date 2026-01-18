import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_triple_match_produces_no_duplicate_items(self):
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <div py:match="div[@id=\'content\']" py:attrs="select(\'@*\')" once="true">\n            <ul id="tabbed_pane" />\n            ${select(\'*\')}\n          </div>\n\n          <body py:match="body" once="true" buffer="false">\n            ${select(\'*|text()\')}\n          </body>\n          <body py:match="body" once="true" buffer="false">\n              ${select(\'*|text()\')}\n          </body>\n\n          <body>\n            <div id="content">\n              <h1>Ticket X</h1>\n            </div>\n          </body>\n        </doc>')
    output = tmpl.generate().render('xhtml', doctype='xhtml')
    matches = re.findall('tabbed_pane', output)
    self.assertNotEqual(None, matches)
    self.assertEqual(1, len(matches))