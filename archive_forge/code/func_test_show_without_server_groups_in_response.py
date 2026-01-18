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
def test_show_without_server_groups_in_response(self):
    out = self.run_command('show 1234', api_version='2.70')[0]
    self.assert_called('GET', '/servers?name=1234', pos=0)
    self.assert_called('GET', '/servers?name=1234', pos=1)
    self.assert_called('GET', '/servers/1234', pos=2)
    self.assert_called('GET', '/v2/images/%s' % FAKE_UUID_2, pos=3)
    self.assertNotIn('server_groups', out)
    self.assertNotIn('a67359fb-d397-4697-88f1-f55e3ee7c499', out)