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
def test_rebuild_user_data_and_unset_user_data(self):
    """Tests that trying to set --user-data and --unset-user-data in the
        same rebuild call fails.
        """
    cmd = 'rebuild sample-server %s --user-data x --user-data-unset' % FAKE_UUID_1
    ex = self.assertRaises(exceptions.CommandError, self.run_command, cmd, api_version='2.57')
    self.assertIn("Cannot specify '--user-data-unset' with '--user-data'.", str(ex))