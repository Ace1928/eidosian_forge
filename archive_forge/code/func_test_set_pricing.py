import sys
import os.path
import unittest
import libcloud.pricing
def test_set_pricing(self):
    self.assertFalse('foo' in libcloud.pricing.PRICING_DATA['compute'])
    libcloud.pricing.set_pricing(driver_type='compute', driver_name='foo', pricing={'foo': 1})
    self.assertTrue('foo' in libcloud.pricing.PRICING_DATA['compute'])