import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
class TestCurrentAttribute(LineTestCase):

    def setUp(self):
        self.func = current_object_attribute

    def test_simple(self):
        self.assertAccess('Object.<attr1|>')
        self.assertAccess('Object.attr1.<attr2|>')
        self.assertAccess('Object.<att|r1>.attr2')
        self.assertAccess('stuff[stuff] + {123: 456} + Object.attr1.<attr2|>')
        self.assertAccess('stuff[asd|fg]')
        self.assertAccess('stuff[asdf[asd|fg]')
        self.assertAccess('Object.attr1.<|attr2>')
        self.assertAccess('Object.<attr1|>.attr2')