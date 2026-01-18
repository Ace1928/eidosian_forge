from test import notPyPy2
import mock
import pytest
from urllib3.exceptions import SNIMissingWarning
from urllib3.util import ssl_
@pytest.mark.parametrize(['has_sni', 'server_hostname', 'should_warn'], [(True, 'www.google.com', False), (True, '127.0.0.1', False), (False, '127.0.0.1', False), (False, 'www.google.com', True), (True, None, False), (False, None, False)])
def test_sni_missing_warning_with_ip_addresses(monkeypatch, has_sni, server_hostname, should_warn):
    monkeypatch.setattr(ssl_, 'HAS_SNI', has_sni)
    sock = mock.Mock()
    context = mock.create_autospec(ssl_.SSLContext)
    with mock.patch('warnings.warn') as warn:
        ssl_.ssl_wrap_socket(sock, server_hostname=server_hostname, ssl_context=context)
    if should_warn:
        assert warn.call_count >= 1
        warnings = [call[0][1] for call in warn.call_args_list]
        assert SNIMissingWarning in warnings
    else:
        assert warn.call_count == 0