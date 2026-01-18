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
@mock.patch.object(openstack_shell.OpenStackImagesShell, 'main')
def test_shell_keyboard_interrupt(self, mock_glance_shell):
    try:
        mock_glance_shell.side_effect = KeyboardInterrupt()
        openstack_shell.main()
    except SystemExit as ex:
        self.assertEqual(130, ex.code)