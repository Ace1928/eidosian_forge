import argparse
from oslo_utils import netutils
import testtools
from neutronclient.common import exceptions
from neutronclient.common import utils
def test_str2dict_optional_keys(self):
    self.assertDictEqual({'key1': 'value1'}, utils.str2dict('key1=value1', optional_keys=['key1', 'key2']))
    self.assertDictEqual({'key1': 'value1', 'key2': 'value2'}, utils.str2dict('key1=value1,key2=value2', optional_keys=['key1', 'key2']))
    e = self.assertRaises(argparse.ArgumentTypeError, utils.str2dict, 'key1=value1,key2=value2,key3=value3', optional_keys=['key1', 'key2'])
    self.assertEqual("Invalid key(s) 'key3' specified. Valid key(s): 'key1, key2'.", str(e))