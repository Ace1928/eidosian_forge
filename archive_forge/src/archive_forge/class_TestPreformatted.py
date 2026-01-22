import logging
import os
from inspect import currentframe, getframeinfo
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import (
class TestPreformatted(unittest.TestCase):

    def test_preformatted_api(self):
        ref = 'a message'
        msg = Preformatted(ref)
        self.assertIs(msg.msg, ref)
        self.assertEqual(str(msg), ref)
        self.assertEqual(repr(msg), "Preformatted('a message')")
        ref = 2
        msg = Preformatted(ref)
        self.assertIs(msg.msg, ref)
        self.assertEqual(str(msg), '2')
        self.assertEqual(repr(msg), 'Preformatted(2)')