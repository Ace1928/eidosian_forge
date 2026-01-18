from boto.exception import JSONResponseError
from boto.opsworks import connect_to_region, regions, RegionInfo
from boto.opsworks.layer1 import OpsWorksConnection
from tests.compat import unittest
def test_validation_errors(self):
    with self.assertRaises(JSONResponseError):
        self.api.create_stack('testbotostack', 'us-east-1', 'badarn', 'badarn2')