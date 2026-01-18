import datetime
import ssl
import sys
import threading
import time
import uuid
import fixtures
import kombu
import kombu.connection
import kombu.transport.memory
from oslo_serialization import jsonutils
from oslo_utils import eventletutils
import testscenarios
import oslo_messaging
from oslo_messaging._drivers import amqpdriver
from oslo_messaging._drivers import common as driver_common
from oslo_messaging._drivers import impl_rabbit as rabbit_driver
from oslo_messaging.exceptions import ConfigurationError
from oslo_messaging.exceptions import MessageDeliveryFailure
from oslo_messaging.tests import utils as test_utils
from oslo_messaging.transport import DriverLoadFailure
from unittest import mock
def test_worker_and_heartbeat(self):
    lock = rabbit_driver.ConnectionLock()
    t1 = self._thread(lock, 1)
    t2 = self._thread(lock, 1, heartbeat=True)
    self.assertAlmostEqual(1, t1(), places=0)
    self.assertAlmostEqual(2, t2(), places=0)