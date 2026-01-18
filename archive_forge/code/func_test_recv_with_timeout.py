import socket
from unittest import mock
from octavia_lib.api.drivers import driver_lib
from octavia_lib.api.drivers import exceptions as driver_exceptions
from octavia_lib.common import constants
from octavia_lib.tests.unit import base
@mock.patch('builtins.memoryview')
def test_recv_with_timeout(self, mock_memoryview):
    mock_socket = mock.MagicMock()
    mock_socket.recv.side_effect = [socket.timeout, b'1', b'\n', b'2', b'3', b'\n']
    mock_socket.recv_into.return_value = 1
    mv_mock = mock.MagicMock()
    mock_memoryview.return_value = mv_mock
    mv_mock.tobytes.return_value = b'"test data"'
    response = self.driver_lib._recv(mock_socket)
    calls = [mock.call(1), mock.call(1)]
    mock_socket.recv.assert_has_calls(calls)
    mock_socket.recv_into.assert_called_once_with(mv_mock.__getitem__(), 1)
    self.assertEqual('test data', response)