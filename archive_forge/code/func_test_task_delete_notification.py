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
def test_task_delete_notification(self):
    now = timeutils.isotime()
    self.task_repo_proxy.remove(self.task_stub_proxy)
    output_logs = self.notifier.get_logs()
    self.assertEqual(1, len(output_logs))
    output_log = output_logs[0]
    self.assertEqual('INFO', output_log['notification_type'])
    self.assertEqual('task.delete', output_log['event_type'])
    self.assertEqual(self.task.task_id, output_log['payload']['id'])
    self.assertEqual(timeutils.isotime(self.task.updated_at), output_log['payload']['updated_at'])
    self.assertEqual(timeutils.isotime(self.task.created_at), output_log['payload']['created_at'])
    self.assertEqual(now, output_log['payload']['deleted_at'])
    if 'location' in output_log['payload']:
        self.fail('Notification contained location field.')
    self.assertNotIn('image_id', output_log['payload'])
    self.assertNotIn('user_id', output_log['payload'])
    self.assertNotIn('request_id', output_log['payload'])