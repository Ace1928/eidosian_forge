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
def test_resource_show_with_attrs(self):
    self.register_keystone_auth_fixture()
    resp_dict = {'resource': {'description': '', 'links': [{'href': 'http://heat.example.com:8004/foo', 'rel': 'self'}, {'href': 'http://heat.example.com:8004/foo2', 'rel': 'resource'}], 'logical_resource_id': 'aResource', 'physical_resource_id': '43b68bae-ed5d-4aed-a99f-0b3d39c2418a', 'required_by': [], 'resource_name': 'aResource', 'resource_status': 'CREATE_COMPLETE', 'resource_status_reason': 'state changed', 'resource_type': 'OS::Nova::Server', 'updated_time': '2014-01-06T16:14:26Z', 'creation_time': '2014-01-06T16:14:26Z', 'attributes': {'attr_a': 'value_of_attr_a', 'attr_b': 'value_of_attr_b'}}}
    stack_id = 'teststack/1'
    resource_name = 'aResource'
    self.mock_request_get('/stacks/%s/resources/%s?with_attr=attr_a&with_attr=attr_b' % (parse.quote(stack_id), parse.quote(encodeutils.safe_encode(resource_name))), resp_dict)
    resource_show_text = self.shell('resource-show {0} {1} --with-attr attr_a --with-attr attr_b'.format(stack_id, resource_name))
    required = ['description', 'links', 'http://heat.example.com:8004/foo[0-9]', 'logical_resource_id', 'aResource', 'physical_resource_id', '43b68bae-ed5d-4aed-a99f-0b3d39c2418a', 'required_by', 'resource_name', 'aResource', 'resource_status', 'CREATE_COMPLETE', 'resource_status_reason', 'state changed', 'resource_type', 'OS::Nova::Server', 'updated_time', '2014-01-06T16:14:26Z']
    for r in required:
        self.assertRegex(resource_show_text, r)