from tests.unit import unittest
from boto.exception import BotoServerError, S3CreateError, JSONResponseError
from httpretty import HTTPretty, httprettified
@httprettified
def test_xml_entity_not_loaded(self):
    xml = '<!DOCTYPE Message [<!ENTITY xxe SYSTEM "http://aws.amazon.com/">]><Message>error:&xxe;</Message>'
    bse = BotoServerError('403', 'Forbidden', body=xml)
    self.assertEqual([], HTTPretty.latest_requests)