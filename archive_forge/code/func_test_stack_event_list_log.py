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
def test_stack_event_list_log(self):
    self.register_keystone_auth_fixture()
    resp_dict = self.event_list_resp_dict(resource_name='aResource', rsrc_eventid1=self.event_id_one, rsrc_eventid2=self.event_id_two)
    stack_id = 'teststack/1'
    self.mock_request_get('/stacks/%s/events?sort_dir=asc' % stack_id, resp_dict)
    event_list_text = self.shell('event-list {0} --format log'.format(stack_id))
    expected = '2013-12-05 14:14:31 [aResource]: CREATE_IN_PROGRESS  state changed\n2013-12-05 14:14:32 [aResource]: CREATE_COMPLETE  state changed\n'
    self.assertEqual(expected, event_list_text)