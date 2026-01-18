import datetime
from unittest import mock
import glance_store
from oslo_config import cfg
import oslo_messaging
import webob
import glance.async_
from glance.common import exception
from glance.common import timeutils
import glance.context
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
from glance.tests import utils
def test_task_failure_notification(self):
    self.task_proxy.fail(message=None)
    output_logs = self.notifier.get_logs()
    self.assertEqual(1, len(output_logs))
    output_log = output_logs[0]
    self.assertEqual('INFO', output_log['notification_type'])
    self.assertEqual('task.failure', output_log['event_type'])
    self.assertEqual(self.task.task_id, output_log['payload']['id'])
    self.assertNotIn('image_id', output_log['payload'])
    self.assertNotIn('user_id', output_log['payload'])
    self.assertNotIn('request_id', output_log['payload'])