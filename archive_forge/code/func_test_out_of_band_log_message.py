import copy
import eventlet
import fixtures
import functools
import logging as pylogging
import platform
import sys
import time
from unittest import mock
from oslo_log import formatters
from oslo_log import log as logging
from oslotest import base
import testtools
from oslo_privsep import capabilities
from oslo_privsep import comm
from oslo_privsep import daemon
from oslo_privsep.tests import testctx
@mock.patch.object(daemon.LOG.logger, 'handle')
def test_out_of_band_log_message(self, handle_mock):
    message = [comm.Message.LOG, self.DICT]
    self.assertEqual(self.client_channel.log, daemon.LOG)
    with mock.patch.object(pylogging, 'makeLogRecord') as mock_make_log, mock.patch.object(daemon.LOG, 'isEnabledFor', return_value=True) as mock_enabled:
        self.client_channel.out_of_band(message)
        mock_make_log.assert_called_once_with(self.EXPECTED)
        handle_mock.assert_called_once_with(mock_make_log.return_value)
        mock_enabled.assert_called_once_with(mock_make_log.return_value.levelno)