import fixtures
import threading
from oslo_config import cfg
import testscenarios
import oslo_messaging
from oslo_messaging.notify import dispatcher
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging.tests import utils as test_utils
from unittest import mock
def _setup_notifier(self, transport, topics=['testtopic'], publisher_id='testpublisher'):
    return oslo_messaging.Notifier(transport, topics=topics, driver='messaging', publisher_id=publisher_id)