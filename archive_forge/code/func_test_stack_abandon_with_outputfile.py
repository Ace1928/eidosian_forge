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
def test_stack_abandon_with_outputfile(self):
    self.register_keystone_auth_fixture()
    abandoned_stack = {'action': 'CREATE', 'status': 'COMPLETE', 'name': 'teststack', 'id': '1', 'resources': {'foo': {'name': 'foo', 'resource_id': 'test-res-id', 'action': 'CREATE', 'status': 'COMPLETE', 'resource_data': {}, 'metadata': {}}}}
    self.mock_request_delete('/stacks/teststack/1/abandon', abandoned_stack)
    with tempfile.NamedTemporaryFile() as file_obj:
        self.shell('stack-abandon teststack/1 -O %s' % file_obj.name)
        result = jsonutils.loads(file_obj.read().decode())
        self.assertEqual(abandoned_stack, result)