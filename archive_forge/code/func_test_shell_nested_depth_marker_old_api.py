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
def test_shell_nested_depth_marker_old_api(self):
    self.register_keystone_auth_fixture()
    stack_id = 'teststack/1'
    nested_id = 'nested/2'
    timestamps = ('2014-01-06T16:14:00Z', '2014-01-06T16:15:00Z', '2014-01-06T16:16:00Z', '2014-01-06T16:17:00Z')
    first_request = '/stacks/%s/events?marker=n_eventid1&nested_depth=1&sort_dir=asc' % stack_id
    self._stub_event_list_response_old_api(stack_id, nested_id, timestamps, first_request)
    list_text = self.shell('event-list %s --nested-depth 1 --marker n_eventid1' % stack_id)
    required = ['id', 'p_eventid2', 'n_eventid1', 'n_eventid2', 'stack_name', 'teststack', 'nested']
    for r in required:
        self.assertRegex(list_text, r)
    self.assertNotRegex(list_text, 'p_eventid1')
    self.assertRegex(list_text, '%s.*\n.*%s.*\n.*%s.*' % timestamps[1:])