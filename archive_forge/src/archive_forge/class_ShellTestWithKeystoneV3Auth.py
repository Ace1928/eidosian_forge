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
class ShellTestWithKeystoneV3Auth(ShellTest):
    auth_env = FAKE_V3_ENV.copy()
    token_url = DEFAULT_V3_AUTH_URL + '/auth/tokens'

    def _assert_auth_plugin_args(self):
        self.assertFalse(self.v2_auth.called)
        body = json.loads(self.v3_auth.last_request.body)
        user = body['auth']['identity']['password']['user']
        self.assertEqual(self.auth_env['OS_USERNAME'], user['name'])
        self.assertEqual(self.auth_env['OS_PASSWORD'], user['password'])
        self.assertEqual(self.auth_env['OS_USER_DOMAIN_NAME'], user['domain']['name'])
        self.assertEqual(self.auth_env['OS_PROJECT_ID'], body['auth']['scope']['project']['id'])

    @mock.patch('glanceclient.v1.client.Client')
    def test_auth_plugin_invocation_with_v1(self, v1_client):
        args = '--os-image-api-version 1 image-list'
        glance_shell = openstack_shell.OpenStackImagesShell()
        glance_shell.main(args.split())
        self.assertEqual(0, self.v3_auth.call_count)

    @mock.patch('glanceclient.v2.client.Client')
    def test_auth_plugin_invocation_with_v2(self, v2_client):
        args = '--os-image-api-version 2 image-list'
        glance_shell = openstack_shell.OpenStackImagesShell()
        glance_shell.main(args.split())
        self.assertEqual(0, self.v3_auth.call_count)

    @mock.patch('keystoneauth1.discover.Discover', side_effect=ks_exc.ClientException())
    def test_api_discovery_failed_with_unversioned_auth_url(self, discover):
        args = '--os-image-api-version 2 --os-auth-url %s image-list' % DEFAULT_UNVERSIONED_AUTH_URL
        glance_shell = openstack_shell.OpenStackImagesShell()
        self.assertRaises(exc.CommandError, glance_shell.main, args.split())

    def test_bash_completion(self):
        stdout, stderr = self.shell('--os-image-api-version 2 bash_completion')
        required = ['--status', 'image-create', 'help', '--size']
        for r in required:
            self.assertIn(r, stdout.split())
        avoided = ['bash_completion', 'bash-completion']
        for r in avoided:
            self.assertNotIn(r, stdout.split())