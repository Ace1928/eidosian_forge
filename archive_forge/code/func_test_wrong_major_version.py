from unittest import mock
import ddt
from cinderclient import api_versions
from cinderclient import exceptions
from cinderclient.tests.unit import test_utils
from cinderclient.tests.unit import utils
from cinderclient.v3 import client
def test_wrong_major_version(self):
    self.assertRaises(exceptions.UnsupportedVersion, api_versions.get_api_version, '4')