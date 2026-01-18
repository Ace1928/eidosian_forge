import hashlib
import io
import logging
import socket
import ssl
import warnings
from itertools import chain
from test import notBrotlipy, onlyBrotlipy, onlyPy2, onlyPy3
import pytest
from mock import Mock, patch
from urllib3 import add_stderr_logger, disable_warnings, util
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.poolmanager import ProxyConfig
from urllib3.util import is_fp_closed
from urllib3.util.connection import _has_ipv6, allowed_gai_family, create_connection
from urllib3.util.proxy import connection_requires_http_tunnel, create_proxy_ssl_context
from urllib3.util.request import _FAILEDTELL, make_headers, rewind_body
from urllib3.util.response import assert_header_parsing
from urllib3.util.ssl_ import (
from urllib3.util.timeout import Timeout
from urllib3.util.url import Url, get_host, parse_url, split_first
from . import clear_warnings
@pytest.mark.parametrize('url, expected_normalized_url', [('HTTP://GOOGLE.COM/MAIL/', 'http://google.com/MAIL/'), ('http://user@domain.com:password@example.com/~tilde@?@', 'http://user%40domain.com:password@example.com/~tilde@?@'), ('HTTP://JeremyCline:Hunter2@Example.com:8080/', 'http://JeremyCline:Hunter2@example.com:8080/'), ('HTTPS://Example.Com/?Key=Value', 'https://example.com/?Key=Value'), ('Https://Example.Com/#Fragment', 'https://example.com/#Fragment'), ('[::1%25]', '[::1%25]'), ('[::Ff%etH0%Ff]/%ab%Af', '[::ff%etH0%FF]/%AB%AF'), ('http://user:pass@[AaAa::Ff%25etH0%Ff]/%ab%Af', 'http://user:pass@[aaaa::ff%etH0%FF]/%AB%AF'), ('http://google.com/p[]?parameter[]="hello"#fragment#', 'http://google.com/p%5B%5D?parameter%5B%5D=%22hello%22#fragment%23'), ('http://google.com/p%5B%5d?parameter%5b%5D=%22hello%22#fragment%23', 'http://google.com/p%5B%5D?parameter%5B%5D=%22hello%22#fragment%23')])
def test_parse_url_normalization(self, url, expected_normalized_url):
    """Assert parse_url normalizes the scheme/host, and only the scheme/host"""
    actual_normalized_url = parse_url(url).url
    assert actual_normalized_url == expected_normalized_url