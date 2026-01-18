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
def test_task_processing_notification_disabled(self):
    self.config(disabled_notifications=['task.processing'])
    self.task_proxy.begin_processing()
    output_logs = self.notifier.get_logs()
    self.assertEqual(0, len(output_logs))