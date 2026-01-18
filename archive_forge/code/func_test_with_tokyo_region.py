import sys
from libcloud.test import unittest
from libcloud.test.compute.test_cloudstack import CloudStackCommonTestCase
from libcloud.compute.drivers.auroracompute import AuroraComputeRegion, AuroraComputeNodeDriver
def test_with_tokyo_region(self):
    driver = self.driver_klass('invalid', 'invalid', region=AuroraComputeRegion.TYO)
    self.assertEqual(driver.path, '/tyo')