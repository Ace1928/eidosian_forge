import json as jsonutils
import logging
import time
import urllib.parse
import uuid
import fixtures
import requests
from requests_mock.contrib import fixture
import testtools
def assertRequestNotInHeader(self, name):
    """Verify that the last request made does not contain a header key.

        The request must have already been made.
        """
    headers = self.requests_mock.last_request.headers
    self.assertNotIn(name, headers)