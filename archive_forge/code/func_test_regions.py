from boto.exception import JSONResponseError
from boto.opsworks import connect_to_region, regions, RegionInfo
from boto.opsworks.layer1 import OpsWorksConnection
from tests.compat import unittest
def test_regions(self):
    response = regions()
    self.assertIsInstance(response[0], RegionInfo)