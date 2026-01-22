import argparse
from oslo_utils import netutils
import testtools
from neutronclient.common import exceptions
from neutronclient.common import utils
class ImportClassTestCase(testtools.TestCase):

    def test_get_client_class_invalid_version(self):
        self.assertRaises(exceptions.UnsupportedVersion, utils.get_client_class, 'image', '2', {'image': '2'})