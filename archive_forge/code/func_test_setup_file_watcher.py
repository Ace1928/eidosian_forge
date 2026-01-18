import datetime
import io
import os
import re
import signal
import sys
import threading
from unittest import mock
import fixtures
import greenlet
from oslotest import base
import oslo_config
from oslo_config import fixture
from oslo_reports import guru_meditation_report as gmr
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import opts
@mock.patch('os.stat')
@mock.patch('time.sleep')
@mock.patch.object(threading.Thread, 'start')
def test_setup_file_watcher(self, mock_thread, mock_sleep, mock_stat):
    version = FakeVersionObj()
    mock_stat.return_value.st_mtime = 3
    gmr.TextGuruMeditation._setup_file_watcher(self.CONF.oslo_reports.file_event_handler, self.CONF.oslo_reports.file_event_handler_interval, version, None, self.CONF.oslo_reports.log_dir)
    mock_stat.assert_called_once_with('/specific/file')
    self.assertEqual(1, mock_thread.called)