import aiohttp  # type: ignore
from aioresponses import aioresponses, core  # type: ignore
import mock
import pytest  # type: ignore
from tests_async.transport import async_compliance
import google.auth._credentials_async
from google.auth.transport import _aiohttp_requests as aiohttp_requests
import google.auth.transport._mtls_helper
class TestAuthorizedSession(object):
    TEST_URL = 'http://example.com/'
    method = 'GET'

    def test_constructor(self):
        authed_session = aiohttp_requests.AuthorizedSession(mock.sentinel.credentials)
        assert authed_session.credentials == mock.sentinel.credentials

    def test_constructor_with_auth_request(self):
        http = mock.create_autospec(aiohttp.ClientSession, instance=True, _auto_decompress=False)
        auth_request = aiohttp_requests.Request(http)
        authed_session = aiohttp_requests.AuthorizedSession(mock.sentinel.credentials, auth_request=auth_request)
        assert authed_session._auth_request == auth_request

    @pytest.mark.asyncio
    async def test_request(self):
        with aioresponses() as mocked:
            credentials = mock.Mock(wraps=CredentialsStub())
            mocked.get(self.TEST_URL, status=200, body='test')
            session = aiohttp_requests.AuthorizedSession(credentials)
            resp = await session.request('GET', 'http://example.com/', headers={'Keep-Alive': 'timeout=5, max=1000', 'fake': b'bytes'})
            assert resp.status == 200
            assert 'test' == await resp.text()
            await session.close()

    @pytest.mark.asyncio
    async def test_ctx(self):
        with aioresponses() as mocked:
            credentials = mock.Mock(wraps=CredentialsStub())
            mocked.get('http://test.example.com', payload=dict(foo='bar'))
            session = aiohttp_requests.AuthorizedSession(credentials)
            resp = await session.request('GET', 'http://test.example.com')
            data = await resp.json()
            assert dict(foo='bar') == data
            await session.close()

    @pytest.mark.asyncio
    async def test_http_headers(self):
        with aioresponses() as mocked:
            credentials = mock.Mock(wraps=CredentialsStub())
            mocked.post('http://example.com', payload=dict(), headers=dict(connection='keep-alive'))
            session = aiohttp_requests.AuthorizedSession(credentials)
            resp = await session.request('POST', 'http://example.com')
            assert resp.headers['Connection'] == 'keep-alive'
            await session.close()

    @pytest.mark.asyncio
    async def test_regexp_example(self):
        with aioresponses() as mocked:
            credentials = mock.Mock(wraps=CredentialsStub())
            mocked.get('http://example.com', status=500)
            mocked.get('http://example.com', status=200)
            session1 = aiohttp_requests.AuthorizedSession(credentials)
            resp1 = await session1.request('GET', 'http://example.com')
            session2 = aiohttp_requests.AuthorizedSession(credentials)
            resp2 = await session2.request('GET', 'http://example.com')
            assert resp1.status == 500
            assert resp2.status == 200
            await session1.close()
            await session2.close()

    @pytest.mark.asyncio
    async def test_request_no_refresh(self):
        credentials = mock.Mock(wraps=CredentialsStub())
        with aioresponses() as mocked:
            mocked.get('http://example.com', status=200)
            authed_session = aiohttp_requests.AuthorizedSession(credentials)
            response = await authed_session.request('GET', 'http://example.com')
            assert response.status == 200
            assert credentials.before_request.called
            assert not credentials.refresh.called
            await authed_session.close()

    @pytest.mark.asyncio
    async def test_request_refresh(self):
        credentials = mock.Mock(wraps=CredentialsStub())
        with aioresponses() as mocked:
            mocked.get('http://example.com', status=401)
            mocked.get('http://example.com', status=200)
            authed_session = aiohttp_requests.AuthorizedSession(credentials)
            response = await authed_session.request('GET', 'http://example.com')
            assert credentials.refresh.called
            assert response.status == 200
            await authed_session.close()