from test import SHORT_TIMEOUT
from test.with_dummyserver import test_connectionpool
import pytest
import dummyserver.testcase
import urllib3.exceptions
import urllib3.util.retry
import urllib3.util.url
from urllib3.contrib import appengine
@pytest.mark.usefixtures('testbed')
class TestGAEConnectionManager(test_connectionpool.TestConnectionPool):

    def setup_method(self, method):
        self.manager = appengine.AppEngineManager()
        self.pool = MockPool(self.host, self.port, self.manager)

    def test_exceptions(self):
        with pytest.raises(urllib3.exceptions.TimeoutError):
            self.pool.request('GET', '/sleep?seconds={}'.format(5 * SHORT_TIMEOUT), timeout=SHORT_TIMEOUT)
        with pytest.raises(urllib3.exceptions.ProtocolError):
            self.manager.request('GET', 'ftp://invalid/url')
        with pytest.raises(urllib3.exceptions.ProtocolError):
            self.manager.request('GET', 'http://0.0.0.0')
        with pytest.raises(appengine.AppEnginePlatformError):
            self.pool.request('GET', '/nbytes?length=33554433')
        body = b'1' * 10485761
        with pytest.raises(appengine.AppEnginePlatformError):
            self.manager.request('POST', '/', body=body)
    test_timeout_float = None
    test_conn_closed = None
    test_nagle = None
    test_socket_options = None
    test_disable_default_socket_options = None
    test_defaults_are_applied = None
    test_tunnel = None
    test_keepalive = None
    test_keepalive_close = None
    test_connection_count = None
    test_connection_count_bigpool = None
    test_for_double_release = None
    test_release_conn_parameter = None
    test_stream_keepalive = None
    test_cleanup_on_connection_error = None
    test_read_chunked_short_circuit = None
    test_read_chunked_on_closed_response = None
    test_timeout = None
    test_connect_timeout = None
    test_connection_error_retries = None
    test_total_timeout = None
    test_none_total_applies_connect = None
    test_timeout_success = None
    test_source_address_error = None
    test_bad_connect = None
    test_partial_response = None
    test_dns_error = None