import fixtures
from unittest import mock
from oslo_config import cfg
from stevedore import driver
import testscenarios
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
from oslo_messaging import transport
def test_noport(self):
    self.assertIsNone(self.url5.hosts[0].port)