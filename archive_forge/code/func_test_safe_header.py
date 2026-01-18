import io
import sys
from unittest import mock
from oslo_utils import encodeutils
from requests import Response
import testtools
from glanceclient.common import utils
def test_safe_header(self):
    self.assertEqual(('somekey', 'somevalue'), utils.safe_header('somekey', 'somevalue'))
    self.assertEqual(('somekey', None), utils.safe_header('somekey', None))
    for sensitive_header in utils.SENSITIVE_HEADERS:
        name, value = utils.safe_header(sensitive_header, encodeutils.safe_encode('somestring'))
        self.assertEqual(sensitive_header, name)
        self.assertTrue(value.startswith('{SHA1}'))
        name, value = utils.safe_header(sensitive_header, None)
        self.assertEqual(sensitive_header, name)
        self.assertIsNone(value)