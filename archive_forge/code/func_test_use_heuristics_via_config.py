import os
import mock
import boto
from boto.pyami.config import Config
from boto.regioninfo import RegionInfo, load_endpoint_json, merge_endpoints
from boto.regioninfo import load_regions, get_regions, connect
from tests.unit import unittest
def test_use_heuristics_via_config(self):
    config = mock.Mock(spec=Config)

    def _getbool(section, name, default=False):
        if section == 'Boto' and name == 'use_endpoint_heuristics':
            return True
        return default
    config.getbool = _getbool
    config.get.return_value = None
    with mock.patch('boto.config', config):
        connection = connect('ec2', 'us-southeast-43', connection_cls=FakeConn, region_cls=TestRegionInfo)
    self.assertIsNotNone(connection)
    self.assertEqual(connection.region.name, 'us-southeast-43')
    expected_endpoint = 'ec2.us-southeast-43.amazonaws.com'
    self.assertEqual(connection.region.endpoint, expected_endpoint)