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
def test_stack_show_without_outputs(self):
    self.register_keystone_auth_fixture()
    resp_dict = {'stack': {'id': '1', 'stack_name': 'teststack', 'stack_status': 'CREATE_COMPLETE', 'creation_time': '2012-10-25T01:58:47Z'}}
    params = {'resolve_outputs': False}
    self.mock_request_get('/stacks/teststack/1', resp_dict, params=params)
    list_text = self.shell('stack-show teststack/1 --no-resolve-outputs')
    required = ['id', 'stack_name', 'stack_status', 'creation_time', 'teststack', 'CREATE_COMPLETE', '2012-10-25T01:58:47Z']
    for r in required:
        self.assertRegex(list_text, r)