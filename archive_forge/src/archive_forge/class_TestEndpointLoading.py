import os
import mock
import boto
from boto.pyami.config import Config
from boto.regioninfo import RegionInfo, load_endpoint_json, merge_endpoints
from boto.regioninfo import load_regions, get_regions, connect
from tests.unit import unittest
class TestEndpointLoading(unittest.TestCase):

    def setUp(self):
        super(TestEndpointLoading, self).setUp()

    def test_load_endpoint_json(self):
        endpoints = load_endpoint_json(boto.ENDPOINTS_PATH)
        self.assertTrue('partitions' in endpoints)

    def test_merge_endpoints(self):
        defaults = {'ec2': {'us-east-1': 'ec2.us-east-1.amazonaws.com', 'us-west-1': 'ec2.us-west-1.amazonaws.com'}}
        additions = {'s3': {'us-east-1': 's3.amazonaws.com'}, 'ec2': {'us-east-1': 'ec2.auto-resolve.amazonaws.com', 'us-west-2': 'ec2.us-west-2.amazonaws.com'}}
        endpoints = merge_endpoints(defaults, additions)
        self.assertEqual(endpoints, {'ec2': {'us-east-1': 'ec2.auto-resolve.amazonaws.com', 'us-west-1': 'ec2.us-west-1.amazonaws.com', 'us-west-2': 'ec2.us-west-2.amazonaws.com'}, 's3': {'us-east-1': 's3.amazonaws.com'}})

    def test_load_regions(self):
        endpoints = load_regions()
        self.assertTrue('us-east-1' in endpoints['ec2'])
        self.assertFalse('test-1' in endpoints['ec2'])
        os.environ['BOTO_ENDPOINTS'] = os.path.join(os.path.dirname(__file__), 'test_endpoints.json')
        self.addCleanup(os.environ.pop, 'BOTO_ENDPOINTS')
        endpoints = load_regions()
        self.assertTrue('us-east-1' in endpoints['ec2'])
        self.assertTrue('test-1' in endpoints['ec2'])
        self.assertEqual(endpoints['ec2']['test-1'], 'ec2.test-1.amazonaws.com')

    def test_get_regions(self):
        ec2_regions = get_regions('ec2')
        self.assertTrue(len(ec2_regions) >= 10)
        west_2 = None
        for region_info in ec2_regions:
            if region_info.name == 'us-west-2':
                west_2 = region_info
                break
        self.assertNotEqual(west_2, None, "Couldn't find the us-west-2 region!")
        self.assertTrue(isinstance(west_2, RegionInfo))
        self.assertEqual(west_2.name, 'us-west-2')
        self.assertEqual(west_2.endpoint, 'ec2.us-west-2.amazonaws.com')
        self.assertEqual(west_2.connection_cls, None)

    def test_get_regions_overrides(self):
        ec2_regions = get_regions('ec2', region_cls=TestRegionInfo, connection_cls=FakeConn)
        self.assertTrue(len(ec2_regions) >= 10)
        west_2 = None
        for region_info in ec2_regions:
            if region_info.name == 'us-west-2':
                west_2 = region_info
                break
        self.assertNotEqual(west_2, None, "Couldn't find the us-west-2 region!")
        self.assertFalse(isinstance(west_2, RegionInfo))
        self.assertTrue(isinstance(west_2, TestRegionInfo))
        self.assertEqual(west_2.name, 'us-west-2')
        self.assertEqual(west_2.endpoint, 'ec2.us-west-2.amazonaws.com')
        self.assertEqual(west_2.connection_cls, FakeConn)