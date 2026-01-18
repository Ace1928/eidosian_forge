from unittest import mock
import ddt
from cinderclient.tests.unit import utils
from cinderclient.v3 import limits
def test_rate(self):
    limit = limits.Limits(None, {'rate': [{'uri': 'uri1', 'regex': 'regex1', 'limit': [{'verb': 'verb1', 'value': 'value1', 'remaining': 'remain1', 'unit': 'unit1', 'next-available': 'next1'}]}, {'uri': 'uri2', 'regex': 'regex2', 'limit': [{'verb': 'verb2', 'value': 'value2', 'remaining': 'remain2', 'unit': 'unit2', 'next-available': 'next2'}]}]}, resp=REQUEST_ID)
    l1 = limits.RateLimit('verb1', 'uri1', 'regex1', 'value1', 'remain1', 'unit1', 'next1')
    l2 = limits.RateLimit('verb2', 'uri2', 'regex2', 'value2', 'remain2', 'unit2', 'next2')
    for item in limit.rate:
        self.assertIn(item, [l1, l2])
    self._assert_request_id(limit)