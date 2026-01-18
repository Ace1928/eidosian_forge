import socket
from unittest import mock
from taskflow.engines.worker_based import proxy
from taskflow import test
from taskflow.utils import threading_utils
def test_start_with_on_wait(self):
    try:
        self.proxy(reset_master_mock=True, on_wait=self.on_wait_mock).start()
    except KeyboardInterrupt:
        pass
    master_calls = self.proxy_start_calls([mock.call.connection.drain_events(timeout=self.de_period), mock.call.on_wait(), mock.call.connection.drain_events(timeout=self.de_period), mock.call.on_wait(), mock.call.connection.drain_events(timeout=self.de_period)], exc_type=KeyboardInterrupt)
    self.master_mock.assert_has_calls(master_calls)