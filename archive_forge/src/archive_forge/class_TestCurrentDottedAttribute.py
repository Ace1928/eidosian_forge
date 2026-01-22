import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
class TestCurrentDottedAttribute(LineTestCase):

    def setUp(self):
        self.func = current_dotted_attribute

    def test_simple(self):
        self.assertAccess('<obj.attr>|')
        self.assertAccess('(<obj.attr>|')
        self.assertAccess('[<obj.attr>|')
        self.assertAccess('m.body[0].value|')
        self.assertAccess('m.body[0].attr.value|')