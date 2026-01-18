import logging
import os
from inspect import currentframe, getframeinfo
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import (
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