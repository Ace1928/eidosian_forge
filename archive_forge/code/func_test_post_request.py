from boto.mws.connection import MWSConnection, api_call_map, destructure_object
from boto.mws.response import (ResponseElement, GetFeedSubmissionListResult,
from boto.exception import BotoServerError
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from mock import MagicMock
def test_post_request(self):
    self.service_connection._mexe = MagicMock(side_effect=BotoServerError(500, 'You request has bee throttled', body=self.default_body_error()))
    with self.assertRaises(BotoServerError) as err:
        self.service_connection.get_lowest_offer_listings_for_asin(MarketplaceId='12345', ASINList='ASIN12345', condition='Any', SellerId='1234', excludeme='True')
        self.assertTrue('throttled' in str(err.reason))
        self.assertEqual(int(err.status), 200)