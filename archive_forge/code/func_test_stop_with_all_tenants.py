import argparse
import base64
import builtins
import collections
import datetime
import io
import os
from unittest import mock
import fixtures
from oslo_utils import timeutils
import testtools
import novaclient
from novaclient import api_versions
from novaclient import base
import novaclient.client
from novaclient import exceptions
import novaclient.shell
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import servers
import novaclient.v2.shell
def test_stop_with_all_tenants(self):
    self.run_command('stop sample-server --all-tenants')
    self.assert_called('GET', '/servers?all_tenants=1&name=sample-server', pos=0)
    self.assert_called('GET', '/servers/1234', pos=1)
    self.assert_called('POST', '/servers/1234/action', {'os-stop': None})