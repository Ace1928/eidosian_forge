import copy
from unittest import mock
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.allocation
class AllocationManagerPaginationTest(testtools.TestCase):

    def setUp(self):
        super(AllocationManagerPaginationTest, self).setUp()
        self.api = utils.FakeAPI(fake_responses_pagination)
        self.mgr = ironicclient.v1.allocation.AllocationManager(self.api)

    def test_allocations_list_limit(self):
        allocations = self.mgr.list(limit=1)
        expect = [('GET', '/v1/allocations/?limit=1', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(allocations))
        expected_resp = ({}, {'next': 'http://127.0.0.1:6385/v1/allocations/?limit=1', 'allocations': [ALLOCATION]})
        self.assertEqual(expected_resp, self.api.responses['/v1/allocations']['GET'])

    def test_allocations_list_marker(self):
        allocations = self.mgr.list(marker=ALLOCATION['uuid'])
        expect = [('GET', '/v1/allocations/?marker=%s' % ALLOCATION['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(allocations))
        expected_resp = ({}, {'next': 'http://127.0.0.1:6385/v1/allocations/?limit=1', 'allocations': [ALLOCATION]})
        self.assertEqual(expected_resp, self.api.responses['/v1/allocations']['GET'])

    def test_allocations_list_pagination_no_limit(self):
        allocations = self.mgr.list(limit=0)
        expect = [('GET', '/v1/allocations', {}, None), ('GET', '/v1/allocations/?limit=1', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(2, len(allocations))
        expected_resp = ({}, {'next': 'http://127.0.0.1:6385/v1/allocations/?limit=1', 'allocations': [ALLOCATION]})
        self.assertEqual(expected_resp, self.api.responses['/v1/allocations']['GET'])