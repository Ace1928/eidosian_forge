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
def test_unshelve_pre_v277_with_az_fails(self):
    """Tests that trying to unshelve with an --availability-zone before
        2.77 is an error.
        """
    self.assertRaises(SystemExit, self.run_command, 'unshelve --availability-zone foo-az sample-server', api_version='2.76')