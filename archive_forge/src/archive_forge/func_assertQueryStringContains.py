import json as jsonutils
import logging
import time
import urllib.parse
import uuid
import fixtures
import requests
from requests_mock.contrib import fixture
import testtools
def assertQueryStringContains(self, **kwargs):
    """Verify the query string contains the expected parameters.

        This method is used to verify that the query string for the most recent
        request made contains all the parameters provided as ``kwargs``, and
        that the value of each parameter contains the value for the kwarg. If
        the value for the kwarg is an empty string (''), then all that's
        verified is that the parameter is present.

        """
    parts = urllib.parse.urlparse(self.requests_mock.last_request.url)
    qs = urllib.parse.parse_qs(parts.query, keep_blank_values=True)
    for k, v in kwargs.items():
        self.assertIn(k, qs)
        self.assertIn(v, qs[k])