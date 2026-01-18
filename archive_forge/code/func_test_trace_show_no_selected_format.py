import io
import json
import os
import sys
from unittest import mock
import ddt
from osprofiler.cmd import shell
from osprofiler import exc
from osprofiler.tests import test
@mock.patch('osprofiler.drivers.redis_driver.Redis.get_report')
def test_trace_show_no_selected_format(self, mock_get):
    mock_get.return_value = self._create_mock_notifications()
    msg = 'You should choose one of the following output formats: json, html or dot.'
    self._test_with_command_error(self._trace_show_cmd(), msg)