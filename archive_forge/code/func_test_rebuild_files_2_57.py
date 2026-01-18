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
def test_rebuild_files_2_57(self):
    """Tests that trying to run the rebuild command with the --file option
        after microversion 2.56 fails.
        """
    testfile = os.path.join(os.path.dirname(__file__), 'testfile.txt')
    cmd = 'rebuild sample-server %s --file /tmp/foo=%s'
    self.assertRaises(SystemExit, self.run_command, cmd % (FAKE_UUID_1, testfile), api_version='2.57')