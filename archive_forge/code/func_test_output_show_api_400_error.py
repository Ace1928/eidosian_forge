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
def test_output_show_api_400_error(self):
    self.register_keystone_auth_fixture()
    output = {'output_key': 'key', 'output_value': 'value', 'description': 'description'}
    stack_dict = {'stack': {'id': '1', 'stack_name': 'teststack', 'stack_status': 'CREATE_COMPLETE', 'creation_time': '2012-10-25T01:58:47Z', 'outputs': [output]}}
    self.mock_request_error('/stacks/teststack/1/outputs/key', 'GET', exc.HTTPNotFound())
    self.mock_request_get('/stacks/teststack/1', stack_dict)
    resp = self.shell('output-show --with-detail teststack/1 key')
    required = ['output_key', 'output_value', 'description', 'key', 'value', 'description']
    for r in required:
        self.assertRegex(resp, r)