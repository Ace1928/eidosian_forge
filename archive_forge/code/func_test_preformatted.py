import logging
import os
from inspect import currentframe, getframeinfo
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import (
def test_preformatted(self):
    self.handler.setFormatter(LegacyPyomoFormatter(base=os.path.dirname(__file__), verbosity=lambda: logger.isEnabledFor(logging.DEBUG)))
    msg = 'This is a long multi-line message that in normal circumstances would be line-wrapped\n        with additional information\n        that normally would be combined.'
    logger.setLevel(logging.WARNING)
    logger.info(msg)
    self.assertEqual(self.stream.getvalue(), '')
    logger.warning(Preformatted(msg))
    ans = msg + '\n'
    self.assertEqual(self.stream.getvalue(), ans)
    logger.warning(msg)
    ans += 'WARNING: This is a long multi-line message that in normal circumstances would\nbe line-wrapped with additional information that normally would be combined.\n'
    self.assertEqual(self.stream.getvalue(), ans)
    logger.setLevel(logging.DEBUG)
    logger.warning(Preformatted(msg))
    ans += msg + '\n'
    self.assertEqual(self.stream.getvalue(), ans)
    logger.warning(msg)
    lineno = getframeinfo(currentframe()).lineno - 1
    ans += 'WARNING: "[base]%stest_log.py", %d, test_preformatted\n' % (os.path.sep, lineno)
    ans += '    This is a long multi-line message that in normal circumstances would be\n    line-wrapped with additional information that normally would be combined.\n'
    self.assertEqual(self.stream.getvalue(), ans)