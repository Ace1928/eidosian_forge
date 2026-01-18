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
def test_usage_list_stitch_together_next_results(self):
    cmd = 'usage-list --start 2000-01-20 --end 2005-02-01'
    stdout, _stderr = self.run_command(cmd, api_version='2.40')
    self.assert_called('GET', '/os-simple-tenant-usage?start=2000-01-20T00:00:00&end=2005-02-01T00:00:00&detailed=1', pos=0)
    markers = ['f079e394-1111-457b-b350-bb5ecc685cdd', 'f079e394-2222-457b-b350-bb5ecc685cdd']
    for pos, marker in enumerate(markers):
        self.assert_called('GET', '/os-simple-tenant-usage?start=2000-01-20T00:00:00&end=2005-02-01T00:00:00&marker=%s&detailed=1' % marker, pos=pos + 1)
    self.assertIn('2       | 50903.53      | 99.42     | 0.00', stdout)