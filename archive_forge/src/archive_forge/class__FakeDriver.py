import fixtures
from unittest import mock
from oslo_config import cfg
from stevedore import driver
import testscenarios
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
from oslo_messaging import transport
class _FakeDriver(object):

    def __init__(self, conf):
        self.conf = conf

    def send(self, *args, **kwargs):
        pass

    def send_notification(self, *args, **kwargs):
        pass

    def listen(self, target, batch_size, batch_timeout):
        pass