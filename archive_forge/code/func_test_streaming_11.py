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
def test_streaming_11(test_client, set_cl):
    """Test serving of streaming responses with HTTP/1.1 protocol."""
    http_connection = test_client.get_connection()
    http_connection.auto_open = False
    http_connection.connect()
    status_line, actual_headers, actual_resp_body = test_client.get('/pov', http_conn=http_connection)
    actual_status = int(status_line[:3])
    assert actual_status == 200
    assert status_line[4:] == 'OK'
    assert actual_resp_body == pov.encode()
    assert not header_exists('Connection', actual_headers)
    if set_cl:
        status_line, actual_headers, actual_resp_body = test_client.get('/stream?set_cl=Yes', http_conn=http_connection)
        assert header_exists('Content-Length', actual_headers)
        assert not header_has_value('Connection', 'close', actual_headers)
        assert not header_exists('Transfer-Encoding', actual_headers)
        assert actual_status == 200
        assert status_line[4:] == 'OK'
        assert actual_resp_body == b'0123456789'
    else:
        status_line, actual_headers, actual_resp_body = test_client.get('/stream', http_conn=http_connection)
        assert not header_exists('Content-Length', actual_headers)
        assert actual_status == 200
        assert status_line[4:] == 'OK'
        assert actual_resp_body == b'0123456789'
        chunked_response = False
        for k, v in actual_headers:
            if k.lower() == 'transfer-encoding':
                if str(v) == 'chunked':
                    chunked_response = True
        if chunked_response:
            assert not header_has_value('Connection', 'close', actual_headers)
        else:
            assert header_has_value('Connection', 'close', actual_headers)
            with pytest.raises(http.client.NotConnected):
                test_client.get('/pov', http_conn=http_connection)
        status_line, actual_headers, actual_resp_body = test_client.head('/stream', http_conn=http_connection)
        assert actual_status == 200
        assert status_line[4:] == 'OK'
        assert actual_resp_body == b''
        assert not header_exists('Transfer-Encoding', actual_headers)
    http_connection.close()