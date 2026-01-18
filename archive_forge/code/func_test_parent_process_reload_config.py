import logging
import multiprocessing
import os
import signal
import socket
import time
import traceback
from unittest import mock
import eventlet
from eventlet import event
from oslotest import base as test_base
from oslo_service import service
from oslo_service.tests import base
from oslo_service.tests import eventlet_service
@mock.patch('signal.alarm')
@mock.patch('os.kill')
@mock.patch('oslo_service.service.ProcessLauncher.stop')
@mock.patch('oslo_service.service.ProcessLauncher._respawn_children')
@mock.patch('oslo_service.service.ProcessLauncher.handle_signal')
@mock.patch('oslo_config.cfg.CONF.log_opt_values')
@mock.patch('oslo_service.systemd.notify_once')
@mock.patch('oslo_config.cfg.CONF.reload_config_files')
@mock.patch('oslo_service.service._is_sighup_and_daemon')
def test_parent_process_reload_config(self, is_sighup_and_daemon_mock, reload_config_files_mock, notify_once_mock, log_opt_values_mock, handle_signal_mock, respawn_children_mock, stop_mock, kill_mock, alarm_mock):
    is_sighup_and_daemon_mock.return_value = True
    respawn_children_mock.side_effect = [None, eventlet.greenlet.GreenletExit()]
    launcher = service.ProcessLauncher(self.conf)
    launcher.sigcaught = 1
    launcher.children = {}
    wrap_mock = mock.Mock()
    launcher.children[222] = wrap_mock
    launcher.wait()
    reload_config_files_mock.assert_called_once_with()
    wrap_mock.service.reset.assert_called_once_with()