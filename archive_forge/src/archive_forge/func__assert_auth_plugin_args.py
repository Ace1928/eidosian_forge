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
def _assert_auth_plugin_args(self):
    self.assertFalse(self.v2_auth.called)
    body = json.loads(self.v3_auth.last_request.body)
    user = body['auth']['identity']['password']['user']
    self.assertEqual(self.auth_env['OS_USERNAME'], user['name'])
    self.assertEqual(self.auth_env['OS_PASSWORD'], user['password'])
    self.assertEqual(self.auth_env['OS_USER_DOMAIN_NAME'], user['domain']['name'])
    self.assertEqual(self.auth_env['OS_PROJECT_ID'], body['auth']['scope']['project']['id'])