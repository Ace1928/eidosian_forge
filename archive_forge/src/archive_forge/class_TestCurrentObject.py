import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
class TestCurrentObject(LineTestCase):

    def setUp(self):
        self.func = current_object

    def test_simple(self):
        self.assertAccess('<Object>.attr1|')
        self.assertAccess('<Object>.|')
        self.assertAccess('Object|')
        self.assertAccess('Object|.')
        self.assertAccess('<Object>.|')
        self.assertAccess('<Object.attr1>.attr2|')
        self.assertAccess('<Object>.att|r1.attr2')
        self.assertAccess('stuff[stuff] + {123: 456} + <Object.attr1>.attr2|')
        self.assertAccess('stuff[asd|fg]')
        self.assertAccess('stuff[asdf[asd|fg]')