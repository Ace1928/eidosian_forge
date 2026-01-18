from oslo_messaging._drivers import common
from oslo_messaging import _utils as utils
from oslo_messaging.tests import utils as test_utils
from unittest import mock
def test_version_is_compatible_no_rev_is_zero(self):
    self.assertTrue(utils.version_is_compatible('1.23.0', '1.23'))