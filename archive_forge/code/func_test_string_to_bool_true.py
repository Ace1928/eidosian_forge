import argparse
from oslo_utils import netutils
import testtools
from neutronclient.common import exceptions
from neutronclient.common import utils
def test_string_to_bool_true(self):
    self.assertTrue(utils.str2bool('true'))