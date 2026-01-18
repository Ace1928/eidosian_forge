import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_recursive_match_2(self):
    """
        When two or more match templates match the same element and also
        themselves output the element they match, avoiding recursion is even
        more complex, but should work.
        """
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <body py:match="body">\n            <div id="header"/>\n            ${select(\'*\')}\n          </body>\n          <body py:match="body">\n            ${select(\'*\')}\n            <div id="footer"/>\n          </body>\n          <body>\n            <h1>Foo</h1>\n          </body>\n        </html>')
    self.assertEqual('<html>\n          <body>\n            <div id="header"/><h1>Foo</h1>\n            <div id="footer"/>\n          </body>\n        </html>', tmpl.generate().render(encoding=None))