from __future__ import division
import json
import os
import pickle
import collections
import contextlib
import warnings
import re
import io
import requests
import pytest
import urllib3
from requests.adapters import HTTPAdapter
from requests.auth import HTTPDigestAuth, _basic_auth_str
from requests.compat import (
from requests.cookies import (
from requests.exceptions import (
from requests.exceptions import SSLError as RequestsSSLError
from requests.models import PreparedRequest
from requests.structures import CaseInsensitiveDict
from requests.sessions import SessionRedirectMixin
from requests.models import urlencode
from requests.hooks import default_hooks
from requests.compat import JSONDecodeError, is_py3, MutableMapping
from .compat import StringIO, u
from .utils import override_environ
from urllib3.util import Timeout as Urllib3Timeout
class TestTimeout:

    def test_stream_timeout(self, httpbin):
        try:
            requests.get(httpbin('delay/10'), timeout=2.0)
        except requests.exceptions.Timeout as e:
            assert 'Read timed out' in e.args[0].args[0]

    @pytest.mark.parametrize('timeout, error_text', (((3, 4, 5), '(connect, read)'), ('foo', 'must be an int, float or None')))
    def test_invalid_timeout(self, httpbin, timeout, error_text):
        with pytest.raises(ValueError) as e:
            requests.get(httpbin('get'), timeout=timeout)
        assert error_text in str(e)

    @pytest.mark.parametrize('timeout', (None, Urllib3Timeout(connect=None, read=None)))
    def test_none_timeout(self, httpbin, timeout):
        """Check that you can set None as a valid timeout value.

        To actually test this behavior, we'd want to check that setting the
        timeout to None actually lets the request block past the system default
        timeout. However, this would make the test suite unbearably slow.
        Instead we verify that setting the timeout to None does not prevent the
        request from succeeding.
        """
        r = requests.get(httpbin('get'), timeout=timeout)
        assert r.status_code == 200

    @pytest.mark.parametrize('timeout', ((None, 0.1), Urllib3Timeout(connect=None, read=0.1)))
    def test_read_timeout(self, httpbin, timeout):
        try:
            requests.get(httpbin('delay/10'), timeout=timeout)
            pytest.fail('The recv() request should time out.')
        except ReadTimeout:
            pass

    @pytest.mark.parametrize('timeout', ((0.1, None), Urllib3Timeout(connect=0.1, read=None)))
    def test_connect_timeout(self, timeout):
        try:
            requests.get(TARPIT, timeout=timeout)
            pytest.fail('The connect() request should time out.')
        except ConnectTimeout as e:
            assert isinstance(e, ConnectionError)
            assert isinstance(e, Timeout)

    @pytest.mark.parametrize('timeout', ((0.1, 0.1), Urllib3Timeout(connect=0.1, read=0.1)))
    def test_total_timeout_connect(self, timeout):
        try:
            requests.get(TARPIT, timeout=timeout)
            pytest.fail('The connect() request should time out.')
        except ConnectTimeout:
            pass

    def test_encoded_methods(self, httpbin):
        """See: https://github.com/psf/requests/issues/2316"""
        r = requests.request(b'GET', httpbin('get'))
        assert r.ok