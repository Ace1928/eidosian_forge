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
def test_volume_update_swap_v285(self):
    """Microversion 2.85, we should also keep the original behavior."""
    self.run_command('volume-update sample-server Work Work', api_version='2.85')
    self.assert_called('PUT', '/servers/1234/os-volume_attachments/Work', {'volumeAttachment': {'volumeId': 'Work'}})