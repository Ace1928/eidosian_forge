import datetime
from unittest import mock
import uuid
from oslo_config import cfg
import oslo_utils.importutils
import glance.async_
from glance.async_ import taskflow_executor
from glance.common import exception
from glance.common import timeutils
from glance import domain
import glance.tests.utils as test_utils
@mock.patch.object(timeutils, 'utcnow')
def test_succeed(self, mock_utcnow):
    mock_utcnow.return_value = datetime.datetime.utcnow()
    self.task.begin_processing()
    self.task.succeed('{"location": "file://home"}')
    self.assertEqual('success', self.task.status)
    self.assertEqual('{"location": "file://home"}', self.task.result)
    self.assertEqual(u'', self.task.message)
    expected = timeutils.utcnow() + datetime.timedelta(hours=CONF.task.task_time_to_live)
    self.assertEqual(expected, self.task.expires_at)