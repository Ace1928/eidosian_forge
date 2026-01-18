import datetime
import logging
import sys
import uuid
import fixtures
from kombu import connection
from oslo_serialization import jsonutils
from oslo_utils import strutils
from oslo_utils import timeutils
from stevedore import dispatch
from stevedore import extension
import testscenarios
import yaml
import oslo_messaging
from oslo_messaging.notify import _impl_log
from oslo_messaging.notify import _impl_test
from oslo_messaging.notify import messaging
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
from unittest import mock
def test_notifier_retry_connection_fails_kafka(self):
    """This test sets a small retry number for notification sending and
        configures a non reachable message bus. The expectation that after the
        configured number of retries the driver gives up the message sending.
        """
    self.config(driver=['messagingv2'], topics=['test-retry'], retry=2, group='oslo_messaging_notifications')
    transport = oslo_messaging.get_notification_transport(self.conf, url='kafka://')
    notifier = oslo_messaging.Notifier(transport)
    notifier.info(test_utils.TestContext(), 'test', {})