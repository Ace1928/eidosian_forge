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
def test_template_show_cfn(self):
    self.register_keystone_auth_fixture()
    template_data = open(os.path.join(TEST_VAR_DIR, 'minimal.template')).read()
    resp_dict = jsonutils.loads(template_data)
    self.mock_request_get('/stacks/teststack/template', resp_dict)
    show_text = self.shell('template-show teststack')
    required = ['{', '  "AWSTemplateFormatVersion": "2010-09-09"', '  "Outputs": {}', '  "Resources": {}', '  "Parameters": {}', '}']
    for r in required:
        self.assertRegex(show_text, r)