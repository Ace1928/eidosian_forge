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
def test_notifier_retry_connection_fails_rabbit(self):
    """This test sets a small retry number for notification sending and
        configures a non reachable message bus. The expectation that after the
        configured number of retries the driver gives up the message sending.
        """
    self.config(driver=['messagingv2'], topics=['test-retry'], retry=2, group='oslo_messaging_notifications')
    self.config(rabbit_retry_backoff=0, group='oslo_messaging_rabbit')
    transport = oslo_messaging.get_notification_transport(self.conf, url='rabbit://')
    notifier = oslo_messaging.Notifier(transport)
    orig_establish_connection = connection.Connection._establish_connection
    calls = []

    def wrapped_establish_connection(*args, **kwargs):
        if len(calls) > 2:
            raise self.TestingException('Connection should only be retried twice due to configuration')
        else:
            calls.append((args, kwargs))
            orig_establish_connection(*args, **kwargs)
    with mock.patch('kombu.connection.Connection._establish_connection', new=wrapped_establish_connection):
        with mock.patch('oslo_messaging.notify.messaging.LOG.exception') as mock_log:
            notifier.info(test_utils.TestContext(), 'test', {})
    self.assertEqual(3, len(calls))
    mock_log.assert_called_once()