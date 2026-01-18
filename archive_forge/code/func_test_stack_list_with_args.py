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
def test_stack_list_with_args(self):
    self.register_keystone_auth_fixture()
    resp_dict = self.stack_list_resp_dict(include_project=True)
    resp = fakes.FakeHTTPResponse(200, 'success, you', {'content-type': 'application/json'}, jsonutils.dumps(resp_dict))
    self.session_jreq_mock.return_value = resp
    self.jreq_mock.return_value = (resp, resp_dict)
    list_text = self.shell('stack-list --limit 2 --marker fake_id --filters=status=COMPLETE --filters=status=FAILED --tags=tag1,tag2 --tags-any=tag3,tag4 --not-tags=tag5,tag6 --not-tags-any=tag7,tag8 --global-tenant --show-deleted --show-hidden --sort-keys=stack_name;creation_time --sort-keys=updated_time --sort-dir=asc')
    required = ['stack_owner', 'project', 'testproject', 'teststack', 'teststack2']
    for r in required:
        self.assertRegex(list_text, r)
    self.assertNotRegex(list_text, 'parent')
    if self.jreq_mock.call_args is None:
        self.assertEqual(1, self.session_jreq_mock.call_count)
        url, method = self.session_jreq_mock.call_args[0]
    else:
        self.assertEqual(1, self.jreq_mock.call_count)
        method, url = self.jreq_mock.call_args[0]
    self.assertEqual('GET', method)
    base_url, query_params = utils.parse_query_url(url)
    self.assertEqual('/stacks', base_url)
    expected_query_dict = {'limit': ['2'], 'status': ['COMPLETE', 'FAILED'], 'marker': ['fake_id'], 'tags': ['tag1,tag2'], 'tags_any': ['tag3,tag4'], 'not_tags': ['tag5,tag6'], 'not_tags_any': ['tag7,tag8'], 'global_tenant': ['True'], 'show_deleted': ['True'], 'show_hidden': ['True'], 'sort_keys': ['stack_name', 'creation_time', 'updated_time'], 'sort_dir': ['asc']}
    self.assertEqual(expected_query_dict, query_params)