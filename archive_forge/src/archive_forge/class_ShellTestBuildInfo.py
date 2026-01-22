import os
import sys
import tempfile
from unittest import mock
import uuid
import fixtures
import io
from keystoneauth1 import fixture as keystone_fixture
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from requests_mock.contrib import fixture as rm_fixture
import testscenarios
import testtools
from urllib import parse
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import http
from heatclient.common import utils
from heatclient import exc
import heatclient.shell
from heatclient.tests.unit import fakes
import heatclient.v1.shell
class ShellTestBuildInfo(ShellBase):

    def setUp(self):
        super(ShellTestBuildInfo, self).setUp()
        self._set_fake_env()

    def _set_fake_env(self):
        """Patch os.environ to avoid required auth info."""
        self.set_fake_env(FAKE_ENV_KEYSTONE_V2)

    def test_build_info(self):
        self.register_keystone_auth_fixture()
        resp_dict = {'build_info': {'api': {'revision': 'api_revision'}, 'engine': {'revision': 'engine_revision'}}}
        self.mock_request_get('/build_info', resp_dict)
        build_info_text = self.shell('build-info')
        required = ['api', 'engine', 'revision', 'api_revision', 'engine_revision']
        for r in required:
            self.assertRegex(build_info_text, r)