from unittest import mock
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
class IOQueueTestCase(test_base.BaseTestCase):

    def setUp(self):
        super(IOQueueTestCase, self).setUp()
        self._mock_queue = mock.Mock()
        queue_patcher = mock.patch.object(ioutils.Queue, 'Queue', new=self._mock_queue)
        queue_patcher.start()
        self.addCleanup(queue_patcher.stop)
        self._mock_client_connected = mock.Mock()
        self._ioqueue = ioutils.IOQueue(self._mock_client_connected)

    def test_get(self):
        self._mock_client_connected.isSet.return_value = True
        self._mock_queue.get.return_value = mock.sentinel.item
        queue_item = self._ioqueue.get(timeout=mock.sentinel.timeout)
        self._mock_queue.get.assert_called_once_with(self._ioqueue, timeout=mock.sentinel.timeout)
        self.assertEqual(mock.sentinel.item, queue_item)

    def _test_get_timeout(self, continue_on_timeout=True):
        self._mock_client_connected.isSet.side_effect = [True, True, False]
        self._mock_queue.get.side_effect = ioutils.Queue.Empty
        queue_item = self._ioqueue.get(timeout=mock.sentinel.timeout, continue_on_timeout=continue_on_timeout)
        expected_calls_number = 2 if continue_on_timeout else 1
        self._mock_queue.get.assert_has_calls([mock.call(self._ioqueue, timeout=mock.sentinel.timeout)] * expected_calls_number)
        self.assertIsNone(queue_item)

    def test_get_continue_on_timeout(self):
        self._test_get_timeout()

    def test_get_break_on_timeout(self):
        self._test_get_timeout(continue_on_timeout=False)

    def test_put(self):
        self._mock_client_connected.isSet.side_effect = [True, True, False]
        self._mock_queue.put.side_effect = ioutils.Queue.Full
        self._ioqueue.put(mock.sentinel.item, timeout=mock.sentinel.timeout)
        self._mock_queue.put.assert_has_calls([mock.call(self._ioqueue, mock.sentinel.item, timeout=mock.sentinel.timeout)] * 2)

    @mock.patch.object(ioutils.IOQueue, 'get')
    def _test_get_burst(self, mock_get, exceeded_max_size=False):
        fake_data = 'fake_data'
        mock_get.side_effect = [fake_data, fake_data, None]
        if exceeded_max_size:
            max_size = 0
        else:
            max_size = constants.SERIAL_CONSOLE_BUFFER_SIZE
        ret_val = self._ioqueue.get_burst(timeout=mock.sentinel.timeout, burst_timeout=mock.sentinel.burst_timeout, max_size=max_size)
        expected_calls = [mock.call(timeout=mock.sentinel.timeout)]
        expected_ret_val = fake_data
        if not exceeded_max_size:
            expected_calls.append(mock.call(timeout=mock.sentinel.burst_timeout, continue_on_timeout=False))
            expected_ret_val += fake_data
        mock_get.assert_has_calls(expected_calls)
        self.assertEqual(expected_ret_val, ret_val)

    def test_get_burst(self):
        self._test_get_burst()

    def test_get_burst_exceeded_size(self):
        self._test_get_burst(exceeded_max_size=True)