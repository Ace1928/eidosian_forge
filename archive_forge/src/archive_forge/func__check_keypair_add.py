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
def _check_keypair_add(self, expected_key_type=None, extra_args='', api_version=None):
    self.run_command('keypair-add %s test' % extra_args, api_version=api_version)
    expected_body = {'keypair': {'name': 'test'}}
    if expected_key_type:
        expected_body['keypair']['type'] = expected_key_type
    self.assert_called('POST', '/os-keypairs', expected_body)