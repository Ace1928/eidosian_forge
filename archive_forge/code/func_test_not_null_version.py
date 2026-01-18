from unittest import mock
import ddt
from cinderclient import api_versions
from cinderclient import exceptions
from cinderclient.tests.unit import test_utils
from cinderclient.tests.unit import utils
from cinderclient.v3 import client
def test_not_null_version(self):
    v = api_versions.APIVersion('1.1')
    self.assertTrue(v)