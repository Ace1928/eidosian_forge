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
class ShellTestEvents(ShellBase):

    def setUp(self):
        super(ShellTestEvents, self).setUp()
        self.set_fake_env(FAKE_ENV_KEYSTONE_V2)
    scenarios = [('integer_id', dict(event_id_one='24', event_id_two='42')), ('uuid_id', dict(event_id_one='3d68809e-c4aa-4dc9-a008-933823d2e44f', event_id_two='43b68bae-ed5d-4aed-a99f-0b3d39c2418a'))]

    def test_event_list(self):
        self.register_keystone_auth_fixture()
        resp_dict = self.event_list_resp_dict(resource_name='aResource', rsrc_eventid1=self.event_id_one, rsrc_eventid2=self.event_id_two)
        stack_id = 'teststack/1'
        resource_name = 'testresource/1'
        self.mock_request_get('/stacks/%s/resources/%s/events?sort_dir=asc' % (parse.quote(stack_id), parse.quote(encodeutils.safe_encode(resource_name))), resp_dict)
        event_list_text = self.shell('event-list {0} --resource {1}'.format(stack_id, resource_name))
        required = ['resource_name', 'id', 'resource_status_reason', 'resource_status', 'event_time', 'aResource', self.event_id_one, self.event_id_two, 'state changed', 'CREATE_IN_PROGRESS', 'CREATE_COMPLETE', '2013-12-05T14:14:31', '2013-12-05T14:14:32']
        for r in required:
            self.assertRegex(event_list_text, r)

    def test_stack_event_list_log(self):
        self.register_keystone_auth_fixture()
        resp_dict = self.event_list_resp_dict(resource_name='aResource', rsrc_eventid1=self.event_id_one, rsrc_eventid2=self.event_id_two)
        stack_id = 'teststack/1'
        self.mock_request_get('/stacks/%s/events?sort_dir=asc' % stack_id, resp_dict)
        event_list_text = self.shell('event-list {0} --format log'.format(stack_id))
        expected = '2013-12-05 14:14:31 [aResource]: CREATE_IN_PROGRESS  state changed\n2013-12-05 14:14:32 [aResource]: CREATE_COMPLETE  state changed\n'
        self.assertEqual(expected, event_list_text)

    def test_event_show(self):
        self.register_keystone_auth_fixture()
        resp_dict = {'event': {'event_time': '2013-12-05T14:14:30Z', 'id': self.event_id_one, 'links': [{'href': 'http://heat.example.com:8004/foo', 'rel': 'self'}, {'href': 'http://heat.example.com:8004/foo2', 'rel': 'resource'}, {'href': 'http://heat.example.com:8004/foo3', 'rel': 'stack'}], 'logical_resource_id': 'aResource', 'physical_resource_id': None, 'resource_name': 'aResource', 'resource_properties': {'admin_user': 'im_powerful', 'availability_zone': 'nova'}, 'resource_status': 'CREATE_IN_PROGRESS', 'resource_status_reason': 'state changed', 'resource_type': 'OS::Nova::Server'}}
        stack_id = 'teststack/1'
        resource_name = 'testresource/1'
        self.mock_request_get('/stacks/%s/resources/%s/events/%s' % (parse.quote(stack_id), parse.quote(encodeutils.safe_encode(resource_name)), parse.quote(self.event_id_one)), resp_dict)
        event_list_text = self.shell('event-show {0} {1} {2}'.format(stack_id, resource_name, self.event_id_one))
        required = ['Property', 'Value', 'event_time', '2013-12-05T14:14:30Z', 'id', self.event_id_one, 'links', 'http://heat.example.com:8004/foo[0-9]', 'logical_resource_id', 'physical_resource_id', 'resource_name', 'aResource', 'resource_properties', 'admin_user', 'availability_zone', 'resource_status', 'CREATE_IN_PROGRESS', 'resource_status_reason', 'state changed', 'resource_type', 'OS::Nova::Server']
        for r in required:
            self.assertRegex(event_list_text, r)