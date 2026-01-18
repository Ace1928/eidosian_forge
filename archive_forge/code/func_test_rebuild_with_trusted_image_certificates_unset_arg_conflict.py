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
def test_rebuild_with_trusted_image_certificates_unset_arg_conflict(self):
    """Tests the error condition that trusted image certs are both unset
        and set via argument during rebuild.
        """
    ex = self.assertRaises(exceptions.CommandError, self.run_command, 'rebuild sample-server %s --trusted-image-certificate-id id1 --trusted-image-certificates-unset' % FAKE_UUID_1, api_version='2.63')
    self.assertIn("Cannot specify '--trusted-image-certificates-unset' with '--trusted-image-certificate-id'", str(ex))