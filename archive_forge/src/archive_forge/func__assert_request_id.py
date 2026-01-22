import json
import os
from unittest import mock
import fixtures
import requests
from requests_mock.contrib import fixture as requests_mock_fixture
import testtools
def _assert_request_id(self, obj, count=1):
    self.assertTrue(hasattr(obj, 'request_ids'))
    self.assertEqual(REQUEST_ID * count, obj.request_ids)