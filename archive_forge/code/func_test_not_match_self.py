import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_not_match_self(self):
    """
        See http://genshi.edgewall.org/ticket/77
        """
    tmpl = MarkupTemplate('<html xmlns="http://www.w3.org/1999/xhtml"\n              xmlns:py="http://genshi.edgewall.org/">\n          <body py:match="body" py:content="select(\'*\')" />\n          <h1 py:match="h1">\n            ${select(\'text()\')}\n            Goodbye!\n          </h1>\n          <body>\n            <h1>Hello!</h1>\n          </body>\n        </html>')
    self.assertEqual('<html xmlns="http://www.w3.org/1999/xhtml">\n          <body><h1>\n            Hello!\n            Goodbye!\n          </h1></body>\n        </html>', tmpl.generate().render(encoding=None))