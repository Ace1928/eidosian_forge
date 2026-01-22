import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
class TestCurrentDict(LineTestCase):

    def setUp(self):
        self.func = current_dict

    def test_simple(self):
        self.assertAccess('asdf|')
        self.assertAccess('asdf|')
        self.assertAccess('<asdf>[|')
        self.assertAccess('<asdf>[|]')
        self.assertAccess('<object.dict>[abc|')
        self.assertAccess('asdf|')