from tests.unit import unittest
from boto.exception import BotoServerError, S3CreateError, JSONResponseError
from httpretty import HTTPretty, httprettified
@httprettified
def test_xmlns_not_loaded(self):
    xml = '<ErrorResponse xmlns="http://elasticloadbalancing.amazonaws.com/doc/2011-11-15/">'
    bse = BotoServerError('403', 'Forbidden', body=xml)
    self.assertEqual([], HTTPretty.latest_requests)