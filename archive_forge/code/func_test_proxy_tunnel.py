import pytest
from urllib3.exceptions import MaxRetryError, NewConnectionError, ProxyError
from urllib3.poolmanager import ProxyManager
from urllib3.util.retry import Retry
from urllib3.util.url import parse_url
from .port_helpers import find_unused_port
def test_proxy_tunnel(self):
    http_url = parse_url('http://example.com')
    https_url = parse_url('https://example.com')
    with ProxyManager('http://proxy:8080') as p:
        assert p._proxy_requires_url_absolute_form(http_url)
        assert p._proxy_requires_url_absolute_form(https_url) is False
    with ProxyManager('https://proxy:8080') as p:
        assert p._proxy_requires_url_absolute_form(http_url)
        assert p._proxy_requires_url_absolute_form(https_url) is False
    with ProxyManager('https://proxy:8080', use_forwarding_for_https=True) as p:
        assert p._proxy_requires_url_absolute_form(http_url)
        assert p._proxy_requires_url_absolute_form(https_url)