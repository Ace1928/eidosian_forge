import pytest
from urllib3.exceptions import MaxRetryError, NewConnectionError, ProxyError
from urllib3.poolmanager import ProxyManager
from urllib3.util.retry import Retry
from urllib3.util.url import parse_url
from .port_helpers import find_unused_port
@pytest.mark.parametrize('proxy_scheme', ['http', 'https'])
def test_proxy_headers(self, proxy_scheme):
    url = 'http://pypi.org/project/urllib3/'
    proxy_url = '{}://something:1234'.format(proxy_scheme)
    with ProxyManager(proxy_url) as p:
        default_headers = {'Accept': '*/*', 'Host': 'pypi.org'}
        headers = p._set_proxy_headers(url)
        assert headers == default_headers
        provided_headers = {'Accept': 'application/json', 'custom': 'header', 'Host': 'test.python.org'}
        headers = p._set_proxy_headers(url, provided_headers)
        assert headers == provided_headers
        provided_headers = {'Accept': 'application/json'}
        expected_headers = provided_headers.copy()
        expected_headers.update({'Host': 'pypi.org:8080'})
        url_with_port = 'http://pypi.org:8080/project/urllib3/'
        headers = p._set_proxy_headers(url_with_port, provided_headers)
        assert headers == expected_headers