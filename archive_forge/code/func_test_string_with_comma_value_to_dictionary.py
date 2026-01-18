import argparse
from oslo_utils import netutils
import testtools
from neutronclient.common import exceptions
from neutronclient.common import utils
def test_string_with_comma_value_to_dictionary(self):
    input_str = 'opt_name=classless-static-route,opt_value=169.254.169.254/32,10.0.0.1'
    expected = {'opt_name': 'classless-static-route', 'opt_value': '169.254.169.254/32,10.0.0.1'}
    self.assertEqual(expected, utils.str2dict(input_str))