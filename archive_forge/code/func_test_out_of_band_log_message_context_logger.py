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
@mock.patch.object(daemon.logging, 'getLogger')
@mock.patch.object(pylogging, 'makeLogRecord')
def test_out_of_band_log_message_context_logger(self, make_log_mock, get_logger_mock):
    logger_name = 'os_brick.privileged'
    context = get_fake_context(conf_attrs={'logger_name': logger_name})
    with mock.patch.object(comm.ClientChannel, '__init__'), mock.patch.object(daemon._ClientChannel, 'exchange_ping'):
        channel = daemon._ClientChannel(mock.ANY, context)
    get_logger_mock.assert_called_once_with(logger_name)
    self.assertEqual(get_logger_mock.return_value, channel.log)
    message = [comm.Message.LOG, self.DICT]
    channel.out_of_band(message)
    make_log_mock.assert_called_once_with(self.EXPECTED)
    channel.log.isEnabledFor.assert_called_once_with(make_log_mock.return_value.levelno)
    channel.log.logger.handle.assert_called_once_with(make_log_mock.return_value)