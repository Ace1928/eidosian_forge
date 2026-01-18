import sys
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.deprecation import (
from pyomo.common.log import LoggingIntercept
from io import StringIO
import logging
def test_relocated_message(self):
    with LoggingIntercept() as LOG:
        self.assertIs(_import_object('oldName', 'pyomo.common.tests.test_deprecated.logger', 'TBD', None, None), logger)
    self.assertRegex(LOG.getvalue().replace('\n', ' '), "DEPRECATED: the 'oldName' attribute has been moved to 'pyomo.common.tests.test_deprecated.logger'")
    with LoggingIntercept() as LOG:
        self.assertIs(_import_object('oldName', 'pyomo.common.tests.test_deprecated._import_object', 'TBD', None, None), _import_object)
    self.assertRegex(LOG.getvalue().replace('\n', ' '), "DEPRECATED: the 'oldName' function has been moved to 'pyomo.common.tests.test_deprecated._import_object'")
    with LoggingIntercept() as LOG:
        self.assertIs(_import_object('oldName', 'pyomo.common.tests.test_deprecated.TestRelocated', 'TBD', None, None), TestRelocated)
    self.assertRegex(LOG.getvalue().replace('\n', ' '), "DEPRECATED: the 'oldName' class has been moved to 'pyomo.common.tests.test_deprecated.TestRelocated'")