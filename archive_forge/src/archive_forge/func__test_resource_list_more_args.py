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
def _test_resource_list_more_args(self, query_args, cmd_args, response_args):
    self.register_keystone_auth_fixture()
    resp_dict = {'resources': [{'resource_name': 'foobar', 'links': [{'href': 'http://heat.example.com:8004/foo/12/resources/foobar', 'rel': 'self'}, {'href': 'http://heat.example.com:8004/foo/12', 'rel': 'stack'}]}]}
    stack_id = 'teststack/1'
    self.mock_request_get('/stacks/%s/resources?%s' % (stack_id, query_args), resp_dict)
    shell_cmd = 'resource-list %s %s' % (stack_id, cmd_args)
    resource_list_text = self.shell(shell_cmd)
    for field in response_args:
        self.assertRegex(resource_list_text, field)