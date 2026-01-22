import sys
import warnings
from oslo_log import log
from sqlalchemy import exc
from testtools import matchers
from keystone.tests import unit
class BaseTestTestCase(unit.BaseTestCase):

    def test_unexpected_exit(self):
        self.assertThat(lambda: sys.exit(), matchers.raises(unit.UnexpectedExit))