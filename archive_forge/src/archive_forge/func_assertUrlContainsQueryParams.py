import os
import random
import unittest
import requests
import requests_mock
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import PY2, httplib, parse_qs, urlparse, urlquote, parse_qsl
from libcloud.common.base import Response
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