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
def test_volume_attach_with_delete_on_termination_v2_79(self):
    out = self.run_command('volume-attach --delete-on-termination sample-server 2 /dev/vdb', api_version='2.79')[0]
    self.assert_called('POST', '/servers/1234/os-volume_attachments', {'volumeAttachment': {'device': '/dev/vdb', 'volumeId': '2', 'delete_on_termination': True}})
    self.assertIn('delete_on_termination', out)