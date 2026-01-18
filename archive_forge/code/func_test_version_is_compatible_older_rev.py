from oslo_messaging._drivers import common
from oslo_messaging import _utils as utils
from oslo_messaging.tests import utils as test_utils
from unittest import mock
def test_version_is_compatible_older_rev(self):
    self.assertTrue(utils.version_is_compatible('1.24', '1.23.1'))