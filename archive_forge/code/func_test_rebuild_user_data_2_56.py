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
def test_rebuild_user_data_2_56(self):
    """Tests that trying to run the rebuild command with the --user-data*
        options before microversion 2.57 fails.
        """
    cmd = 'rebuild sample-server %s --user-data test' % FAKE_UUID_1
    self.assertRaises(SystemExit, self.run_command, cmd, api_version='2.56')
    cmd = 'rebuild sample-server %s --user-data-unset' % FAKE_UUID_1
    self.assertRaises(SystemExit, self.run_command, cmd, api_version='2.56')