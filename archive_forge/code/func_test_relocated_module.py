import sys
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.deprecation import (
from pyomo.common.log import LoggingIntercept
from io import StringIO
import logging
def test_relocated_module(self):
    with LoggingIntercept() as LOG:
        from pyomo.common.tests.relo_mod import ReloClass
    self.assertRegex(LOG.getvalue().replace('\n', ' '), "DEPRECATED: The 'pyomo\\.common\\.tests\\.relo_mod' module has been moved to 'pyomo\\.common\\.tests\\.relo_mod_new'. Please update your import. \\(deprecated in 1\\.2\\) \\(called from .*test_deprecated\\.py")
    with LoggingIntercept() as LOG:
        import pyomo.common.tests.relo_mod as relo
    self.assertEqual(LOG.getvalue(), '')
    import pyomo.common.tests.relo_mod_new as relo_new
    self.assertIs(relo, relo_new)
    self.assertEqual(relo.RELO_ATTR, 42)
    self.assertIs(ReloClass, relo_new.ReloClass)