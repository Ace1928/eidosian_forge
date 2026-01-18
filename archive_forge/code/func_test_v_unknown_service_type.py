import argparse
import io
import re
import sys
from unittest import mock
import ddt
import fixtures
from keystoneauth1 import fixture
import requests_mock
from testtools import matchers
from novaclient import api_versions
import novaclient.client
from novaclient import exceptions
import novaclient.shell
from novaclient.tests.unit import fake_actions_module
from novaclient.tests.unit import utils
def test_v_unknown_service_type(self):
    self.assertRaises(exceptions.UnsupportedVersion, self._test_service_type, 'unknown', 'compute', self.mock_client)