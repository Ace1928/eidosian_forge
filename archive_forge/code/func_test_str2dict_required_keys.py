import argparse
from oslo_utils import netutils
import testtools
from neutronclient.common import exceptions
from neutronclient.common import utils
def test_str2dict_required_keys(self):
    self.assertDictEqual({'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}, utils.str2dict('key1=value1,key2=value2,key3=value3', required_keys=['key1', 'key2'], optional_keys=['key3']))
    self.assertDictEqual({'key1': 'value1', 'key2': 'value2'}, utils.str2dict('key1=value1,key2=value2', required_keys=['key1', 'key2']))
    e = self.assertRaises(argparse.ArgumentTypeError, utils.str2dict, 'key1=value1', required_keys=['key1', 'key2'])
    self.assertEqual("Required key(s) 'key2' not specified.", str(e))