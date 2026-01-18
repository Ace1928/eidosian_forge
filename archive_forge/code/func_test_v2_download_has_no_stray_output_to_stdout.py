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
def test_v2_download_has_no_stray_output_to_stdout(self):
    """Ensure no stray print statements corrupt the image"""
    saved_stdout = sys.stdout
    try:
        sys.stdout = output = testutils.FakeNoTTYStdout()
        id = image_show_fixture['id']
        headers = {'Content-Length': '4', 'Content-type': 'application/octet-stream'}
        fake = testutils.FakeResponse(headers, io.StringIO('DATA'))
        self.requests = self.useFixture(rm_fixture.Fixture())
        self.requests.get('http://example.com/v2/images/%s/file' % id, headers=headers, raw=fake)
        self.requests.get('http://example.com/v2/images/%s' % id, headers={'Content-type': 'application/json'}, json=image_show_fixture)
        shell = openstack_shell.OpenStackImagesShell()
        argstr = '--os-image-api-version 2 --os-auth-token faketoken --os-image-url http://example.com image-download %s' % id
        shell.main(argstr.split())
        self.assertEqual('DATA', output.getvalue())
    finally:
        sys.stdout = saved_stdout