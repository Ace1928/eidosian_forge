import argparse
from collections import OrderedDict
import hashlib
import io
import logging
import os
import sys
import traceback
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import exceptions as ks_exc
from keystoneauth1 import fixture as ks_fixture
from requests_mock.contrib import fixture as rm_fixture
from glanceclient.common import utils
from glanceclient import exc
from glanceclient import shell as openstack_shell
from glanceclient.tests.unit.v2.fixtures import image_show_fixture
from glanceclient.tests.unit.v2.fixtures import image_versions_fixture
from glanceclient.tests import utils as testutils
from glanceclient.v2 import schemas as schemas
import json
@mock.patch.object(openstack_shell.OpenStackImagesShell, 'get_subcommand_parser')
def test_shell_import_error_with_mesage(self, mock_parser):
    msg = 'Unable to import module xxx'
    mock_parser.side_effect = ImportError('%s' % msg)
    shell = openstack_shell.OpenStackImagesShell()
    argstr = '--os-image-api-version 2 image-list'
    try:
        shell.main(argstr.split())
        self.fail('No import error returned')
    except ImportError as e:
        self.assertEqual(msg, str(e))