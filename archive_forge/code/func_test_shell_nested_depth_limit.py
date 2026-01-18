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
def test_shell_nested_depth_limit(self):
    self.register_keystone_auth_fixture()
    stack_id = 'teststack/1'
    nested_events = self._nested_events()
    ev_resp_dict = {'events': nested_events[:2]}
    url = '/stacks/%s/events?limit=2&nested_depth=1&sort_dir=asc' % stack_id
    self.mock_request_get(url, ev_resp_dict)
    list_text = self.shell('event-list %s --nested-depth 1 --format log --limit 2' % stack_id)
    self.assertEqual('2014-01-06 16:14:00Z [the_stack]: CREATE_IN_PROGRESS  Stack CREATE started\n2014-01-06 16:15:00Z [nested_stack]: CREATE_IN_PROGRESS  Stack CREATE started\n', list_text)