import os
import random
import unittest
import requests
import requests_mock
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import PY2, httplib, parse_qs, urlparse, urlquote, parse_qsl
from libcloud.common.base import Response
class MockHttp(LibcloudConnection):
    """
    A mock HTTP client/server suitable for testing purposes. This replaces
    `HTTPConnection` by implementing its API and returning a mock response.

    Define methods by request path, replacing slashes (/) with underscores (_).
    Each of these mock methods should return a tuple of:

        (int status, str body, dict headers, str reason)
    """
    type = None
    use_param = None
    test = None
    proxy_url = None

    def __init__(self, *args, **kwargs):
        if isinstance(self, unittest.TestCase):
            unittest.TestCase.__init__(self, '__init__')
        super().__init__(*args, **kwargs)

    def _get_request(self, method, url, body=None, headers=None):
        parsed = urlparse.urlparse(url)
        _, _, path, _, query, _ = parsed
        qs = parse_qs(query)
        if path.endswith('/'):
            path = path[:-1]
        meth_name = self._get_method_name(type=self.type, use_param=self.use_param, qs=qs, path=path)
        meth = getattr(self, meth_name.replace('%', '_'))
        if self.test and isinstance(self.test, LibcloudTestCase):
            self.test._add_visited_url(url=url)
            self.test._add_executed_mock_method(method_name=meth_name)
        return meth(method, url, body, headers)

    def request(self, method, url, body=None, headers=None, raw=False, stream=False):
        headers = self._normalize_headers(headers=headers)
        r_status, r_body, r_headers, r_reason = self._get_request(method, url, body, headers)
        if r_body is None:
            r_body = ''
        url = urlquote(url)
        with requests_mock.mock() as m:
            m.register_uri(method, url, text=r_body, reason=r_reason, headers=r_headers, status_code=r_status)
            try:
                super().request(method=method, url=url, body=body, headers=headers, raw=raw, stream=stream)
            except requests_mock.exceptions.NoMockAddress as nma:
                raise AttributeError('Failed to mock out URL {} - {}'.format(url, nma.request.url))

    def prepared_request(self, method, url, body=None, headers=None, raw=False, stream=False):
        headers = self._normalize_headers(headers=headers)
        r_status, r_body, r_headers, r_reason = self._get_request(method, url, body, headers)
        with requests_mock.mock() as m:
            m.register_uri(method, url, text=r_body, reason=r_reason, headers=r_headers, status_code=r_status)
            super().prepared_request(method=method, url=url, body=body, headers=headers, raw=raw, stream=stream)

    def _example(self, method, url, body, headers):
        """
        Return a simple message and header, regardless of input.
        """
        return (httplib.OK, 'Hello World!', {'X-Foo': 'libcloud'}, httplib.responses[httplib.OK])

    def _example_fail(self, method, url, body, headers):
        return (httplib.FORBIDDEN, 'Oh No!', {'X-Foo': 'fail'}, httplib.responses[httplib.FORBIDDEN])

    def _get_method_name(self, type, use_param, qs, path):
        path = path.split('?')[0]
        meth_name = path.replace('/', '_').replace('.', '_').replace('-', '_').replace('~', '%7E')
        if type:
            meth_name = '{}_{}'.format(meth_name, self.type)
        if use_param and use_param in qs:
            param = qs[use_param][0].replace('.', '_').replace('-', '_')
            meth_name = '{}_{}'.format(meth_name, param)
        if meth_name == '':
            meth_name = 'root'
        return meth_name

    def assertUrlContainsQueryParams(self, url, expected_params, strict=False):
        """
        Assert that provided url contains provided query parameters.

        :param url: URL to assert.
        :type url: ``str``

        :param expected_params: Dictionary of expected query parameters.
        :type expected_params: ``dict``

        :param strict: Assert that provided url contains only expected_params.
                       (defaults to ``False``)
        :type strict: ``bool``
        """
        question_mark_index = url.find('?')
        if question_mark_index != -1:
            url = url[question_mark_index + 1:]
        params = dict(parse_qsl(url))
        if strict:
            assert params == expected_params
        else:
            for key, value in expected_params.items():
                assert key in params
                assert params[key] == value