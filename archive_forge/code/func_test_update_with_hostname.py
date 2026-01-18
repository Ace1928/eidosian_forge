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
def test_update_with_hostname(self):
    self.run_command('update --hostname new-hostname sample-server', api_version='2.90')
    expected_put_body = {'server': {'hostname': 'new-hostname'}}
    self.assert_called('GET', '/servers/1234', pos=-2)
    self.assert_called('PUT', '/servers/1234', expected_put_body, pos=-1)