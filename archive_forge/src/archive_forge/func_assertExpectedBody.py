import base64
import copy
from unittest import mock
from urllib import parse as urlparse
from oslo_utils import uuidutils
from osprofiler import _utils as osprofiler_utils
import osprofiler.profiler
from mistralclient.api import httpclient
from mistralclient.tests.unit import base
def assertExpectedBody(self):
    text = self.requests_mock.last_request.text
    form = urlparse.parse_qs(text, strict_parsing=True)
    self.assertEqual(len(EXPECTED_BODY), len(form))
    for k, v in EXPECTED_BODY.items():
        self.assertEqual([str(v)], form[k])
    return form