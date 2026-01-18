import re
import unittest
from oslo_config import types
def test_valid_hostname(self):
    self.assertConvertedEqual('cell1.nova.site1')
    self.assertConvertedEqual('site01001')
    self.assertConvertedEqual('home-site-here.org.com')
    self.assertConvertedEqual('localhost')
    self.assertConvertedEqual('3com.com')
    self.assertConvertedEqual('10.org')
    self.assertConvertedEqual('10ab.10ab')
    self.assertConvertedEqual('ab-c.com')
    self.assertConvertedEqual('abc.com-org')
    self.assertConvertedEqual('abc.0-0')