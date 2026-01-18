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
@mock.patch('sys.stdout', io.StringIO())
@mock.patch('sys.stderr', io.StringIO())
@mock.patch('sys.argv', ['glance', '--debug', 'help', 'foofoo'])
def test_stacktrace_when_debug_enabled(self):
    with mock.patch.object(traceback, 'print_exc') as mock_print_exc:
        try:
            openstack_shell.main()
        except SystemExit:
            pass
        self.assertTrue(mock_print_exc.called)