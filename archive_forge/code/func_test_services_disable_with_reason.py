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
def test_services_disable_with_reason(self):
    self.run_command('service-disable host1 --reason no_reason')
    body = {'host': 'host1', 'binary': 'nova-compute', 'disabled_reason': 'no_reason'}
    self.assert_called('PUT', '/os-services/disable-log-reason', body)