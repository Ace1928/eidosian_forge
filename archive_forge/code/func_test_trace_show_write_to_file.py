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
def test_trace_show_write_to_file(self, mock_get):
    notifications = self._create_mock_notifications()
    mock_get.return_value = notifications
    with mock.patch('osprofiler.cmd.commands.open', mock.mock_open(), create=True) as mock_open:
        self.run_command("%s --out='/file'" % self._trace_show_cmd(format_='json'))
        output = mock_open.return_value.__enter__.return_value
        output.write.assert_called_once_with(json.dumps(notifications, indent=2, separators=(',', ': ')))