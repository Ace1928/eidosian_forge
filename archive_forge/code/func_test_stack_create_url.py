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
def test_stack_create_url(self):
    self.register_keystone_auth_fixture()
    url_content = io.StringIO('{"AWSTemplateFormatVersion" : "2010-09-09"}')
    self.useFixture(fixtures.MockPatchObject(request, 'urlopen', return_value=url_content))
    expected_data = {'files': {}, 'disable_rollback': True, 'stack_name': 'teststack', 'environment': {}, 'template': {'AWSTemplateFormatVersion': '2010-09-09'}, 'parameters': {'DBUsername': 'wp', 'KeyName': 'heat_key', 'LinuxDistribution': 'F17"', '"InstanceType': 'm1.large', 'DBPassword': 'verybadpassword'}}
    self.mock_request_post('/stacks', None, data=expected_data, status_code=201, req_headers=True)
    self.mock_stack_list()
    create_text = self.shell('stack-create teststack --template-url=http://no.where/minimal.template --parameters="InstanceType=m1.large;DBUsername=wp;DBPassword=verybadpassword;KeyName=heat_key;LinuxDistribution=F17"')
    required = ['stack_name', 'id', 'teststack2', '2']
    for r in required:
        self.assertRegex(create_text, r)
    request.urlopen.assert_called_once_with('http://no.where/minimal.template')