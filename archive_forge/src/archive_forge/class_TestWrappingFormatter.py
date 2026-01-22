import logging
import os
from inspect import currentframe, getframeinfo
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import (
class TestWrappingFormatter(unittest.TestCase):

    def setUp(self):
        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        logger.addHandler(self.handler)

    def tearDown(self):
        logger.removeHandler(self.handler)

    def test_style_options(self):
        ans = ''
        self.handler.setFormatter(WrappingFormatter(style='%'))
        logger.warning('(warn)')
        ans += 'WARNING: (warn)\n'
        self.assertEqual(self.stream.getvalue(), ans)
        self.handler.setFormatter(WrappingFormatter(style='$'))
        logger.warning('(warn)')
        ans += 'WARNING: (warn)\n'
        self.assertEqual(self.stream.getvalue(), ans)
        self.handler.setFormatter(WrappingFormatter(style='{'))
        logger.warning('(warn)')
        ans += 'WARNING: (warn)\n'
        self.assertEqual(self.stream.getvalue(), ans)
        with self.assertRaisesRegex(ValueError, 'unrecognized style flag "s"'):
            WrappingFormatter(style='s')