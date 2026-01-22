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
class MockShellTestToken(MockShellTestUserPass):

    def setUp(self):
        self.token = 'a_token'
        super(MockShellTestToken, self).setUp()

    def _set_fake_env(self):
        fake_env = {'OS_AUTH_TOKEN': self.token, 'OS_TENANT_ID': 'tenant_id', 'OS_AUTH_URL': BASE_URL, 'OS_USERNAME': 'username', 'OS_PASSWORD': 'password'}
        self.set_fake_env(fake_env)