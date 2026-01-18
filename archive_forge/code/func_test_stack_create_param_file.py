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
def test_stack_create_param_file(self):
    self.register_keystone_auth_fixture()
    self.mock_request_post('/stacks', None, data=mock.ANY, status_code=201, req_headers=True)
    self.mock_stack_list()
    self.useFixture(fixtures.MockPatchObject(utils, 'read_url_content', return_value='xxxxxx'))
    url = 'file://' + request.pathname2url('%s/private_key.env' % TEST_VAR_DIR)
    template_file = os.path.join(TEST_VAR_DIR, 'minimal.template')
    create_text = self.shell('stack-create teststack --template-file=%s --parameter-file private_key=private_key.env --parameters="InstanceType=m1.large;DBUsername=wp;DBPassword=verybadpassword;KeyName=heat_key;LinuxDistribution=F17"' % template_file)
    required = ['stack_name', 'id', 'teststack', '1']
    for r in required:
        self.assertRegex(create_text, r)
    utils.read_url_content.assert_called_once_with(url)