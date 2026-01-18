import io
import json
import os
import sys
from unittest import mock
import ddt
from osprofiler.cmd import shell
from osprofiler import exc
from osprofiler.tests import test
@mock.patch('sys.stdout', io.StringIO())
@mock.patch('osprofiler.drivers.redis_driver.Redis.get_report')
def test_trace_show_in_json(self, mock_get):
    notifications = self._create_mock_notifications()
    mock_get.return_value = notifications
    self.run_command(self._trace_show_cmd(format_='json'))
    self.assertEqual('%s\n' % json.dumps(notifications, indent=2, separators=(',', ': ')), sys.stdout.getvalue())