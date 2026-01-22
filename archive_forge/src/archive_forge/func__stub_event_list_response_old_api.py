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
def _stub_event_list_response_old_api(self, stack_id, nested_id, timestamps, first_request):
    ev_resp_dict = {'events': [{'id': 'p_eventid1', 'event_time': timestamps[0]}, {'id': 'p_eventid2', 'event_time': timestamps[3]}]}
    self.mock_request_get(first_request, ev_resp_dict)
    self.mock_request_get('/stacks/%s/events?sort_dir=asc' % stack_id, ev_resp_dict)
    res_resp_dict = {'resources': [{'links': [{'href': 'http://heat/foo', 'rel': 'self'}, {'href': 'http://heat/foo2', 'rel': 'resource'}, {'href': 'http://heat/%s' % nested_id, 'rel': 'nested'}], 'resource_type': 'OS::Nested::Foo'}]}
    self.mock_request_get('/stacks/%s/resources' % stack_id, res_resp_dict)
    nev_resp_dict = {'events': [{'id': 'n_eventid1', 'event_time': timestamps[1]}, {'id': 'n_eventid2', 'event_time': timestamps[2]}]}
    self.mock_request_get('/stacks/%s/events?sort_dir=asc' % nested_id, nev_resp_dict)