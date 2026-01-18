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
def test_template_validate(self):
    self.register_keystone_auth_fixture()
    resp_dict = {'heat_template_version': '2013-05-23', 'parameters': {}, 'resources': {}, 'outputs': {}}
    self.mock_request_post('/validate', resp_dict, data=mock.ANY)
    template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
    cmd = 'template-validate -f %s -P foo=bar' % template_file
    show_text = self.shell(cmd)
    required = ['heat_template_version', 'outputs', 'parameters', 'resources']
    for r in required:
        self.assertRegex(show_text, r)