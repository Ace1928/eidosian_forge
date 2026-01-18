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
def test_output_show_all(self):
    self.register_keystone_auth_fixture()
    resp_dict = {'outputs': [{'output_key': 'key', 'description': 'description'}]}
    resp_dict1 = {'output': {'output_key': 'key', 'output_value': 'value', 'description': 'description'}}
    self.mock_request_get('/stacks/teststack/1/outputs', resp_dict)
    self.mock_request_get('/stacks/teststack/1/outputs/key', resp_dict1)
    list_text = self.shell('output-show --with-detail teststack/1 --all')
    required = ['output_key', 'output_value', 'description', 'key', 'value', 'description']
    for r in required:
        self.assertRegex(list_text, r)