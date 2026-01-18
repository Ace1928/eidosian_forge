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
def test_usage_no_tenant(self):
    self.run_command('usage --start 2000-01-20 --end 2005-02-01')
    self.assert_called('GET', '/os-simple-tenant-usage/tenant_id?' + 'start=2000-01-20T00:00:00&' + 'end=2005-02-01T00:00:00')