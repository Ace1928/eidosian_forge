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
def test_quota_class_update(self):
    args = ('--instances', '--cores', '--ram', '--floating-ips', '--fixed-ips', '--metadata-items', '--injected-files', '--injected-file-content-bytes', '--injected-file-path-bytes', '--key-pairs', '--security-groups', '--security-group-rules', '--server-groups', '--server-group-members')
    for arg in args:
        self.run_command('quota-class-update 97f4c221bff44578b0300df4ef119353 %s=5' % arg)
        request_param = arg[2:].replace('-', '_')
        body = {'quota_class_set': {request_param: 5}}
        self.assert_called('PUT', '/os-quota-class-sets/97f4c221bff44578b0300df4ef119353', body)