import errno
import socket
import time
import logging
import traceback as traceback_
from collections import namedtuple
import http.client
import urllib.request
import pytest
from jaraco.text import trim, unwrap
from cheroot.test import helper, webtest
from cheroot._compat import IS_CI, IS_MACOS, IS_PYPY, IS_WINDOWS
import cheroot.server
@pytest.mark.parametrize('set_cl', (False, True))
def test_streaming_10(test_client, set_cl):
    """Test serving of streaming responses with HTTP/1.0 protocol."""
    original_server_protocol = test_client.server_instance.protocol
    test_client.server_instance.protocol = 'HTTP/1.0'
    http_connection = test_client.get_connection()
    http_connection.auto_open = False
    http_connection.connect()
    status_line, actual_headers, actual_resp_body = test_client.get('/pov', http_conn=http_connection, headers=[('Connection', 'Keep-Alive')], protocol='HTTP/1.0')
    actual_status = int(status_line[:3])
    assert actual_status == 200
    assert status_line[4:] == 'OK'
    assert actual_resp_body == pov.encode()
    assert header_has_value('Connection', 'Keep-Alive', actual_headers)
    if set_cl:
        status_line, actual_headers, actual_resp_body = test_client.get('/stream?set_cl=Yes', http_conn=http_connection, headers=[('Connection', 'Keep-Alive')], protocol='HTTP/1.0')
        actual_status = int(status_line[:3])
        assert actual_status == 200
        assert status_line[4:] == 'OK'
        assert actual_resp_body == b'0123456789'
        assert header_exists('Content-Length', actual_headers)
        assert header_has_value('Connection', 'Keep-Alive', actual_headers)
        assert not header_exists('Transfer-Encoding', actual_headers)
    else:
        status_line, actual_headers, actual_resp_body = test_client.get('/stream', http_conn=http_connection, headers=[('Connection', 'Keep-Alive')], protocol='HTTP/1.0')
        actual_status = int(status_line[:3])
        assert actual_status == 200
        assert status_line[4:] == 'OK'
        assert actual_resp_body == b'0123456789'
        assert not header_exists('Content-Length', actual_headers)
        assert not header_has_value('Connection', 'Keep-Alive', actual_headers)
        assert not header_exists('Transfer-Encoding', actual_headers)
        with pytest.raises(http.client.NotConnected):
            test_client.get('/pov', http_conn=http_connection, protocol='HTTP/1.0')
    test_client.server_instance.protocol = original_server_protocol
    http_connection.close()