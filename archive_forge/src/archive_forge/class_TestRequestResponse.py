import aiohttp  # type: ignore
from aioresponses import aioresponses, core  # type: ignore
import mock
import pytest  # type: ignore
from tests_async.transport import async_compliance
import google.auth._credentials_async
from google.auth.transport import _aiohttp_requests as aiohttp_requests
import google.auth.transport._mtls_helper
class TestRequestResponse(async_compliance.RequestResponseTests):

    def make_request(self):
        return aiohttp_requests.Request()

    def make_with_parameter_request(self):
        http = aiohttp.ClientSession(auto_decompress=False)
        return aiohttp_requests.Request(http)

    def test_unsupported_session(self):
        http = aiohttp.ClientSession(auto_decompress=True)
        with pytest.raises(ValueError):
            aiohttp_requests.Request(http)

    def test_timeout(self):
        http = mock.create_autospec(aiohttp.ClientSession, instance=True, _auto_decompress=False)
        request = aiohttp_requests.Request(http)
        request(url='http://example.com', method='GET', timeout=5)