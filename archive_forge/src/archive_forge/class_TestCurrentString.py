import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
class TestCurrentString(LineTestCase):

    def setUp(self):
        self.func = current_string

    def test_closed(self):
        self.assertAccess('"<as|df>"')
        self.assertAccess('"<asdf|>"')
        self.assertAccess('"<|asdf>"')
        self.assertAccess("'<asdf|>'")
        self.assertAccess("'<|asdf>'")
        self.assertAccess("'''<asdf|>'''")
        self.assertAccess('"""<asdf|>"""')
        self.assertAccess('asdf.afd("a") + "<asdf|>"')

    def test_open(self):
        self.assertAccess('"<as|df>')
        self.assertAccess('"<asdf|>')
        self.assertAccess('"<|asdf>')
        self.assertAccess("'<asdf|>")
        self.assertAccess("'<|asdf>")
        self.assertAccess("'''<asdf|>")
        self.assertAccess('"""<asdf|>')
        self.assertAccess('asdf.afd("a") + "<asdf|>')