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
@mock.patch.object(gmr.TextGuruMeditation, '_setup_file_watcher')
def test_register_autorun_without_signals(self, mock_setup_fh):
    version = FakeVersionObj()
    gmr.TextGuruMeditation.setup_autorun(version, conf=self.CONF)
    mock_setup_fh.assert_called_once_with('/specific/file', 10, version, None, '/var/fake_log')