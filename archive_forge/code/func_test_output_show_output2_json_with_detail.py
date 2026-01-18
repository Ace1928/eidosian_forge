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
def test_output_show_output2_json_with_detail(self):
    self.register_keystone_auth_fixture()
    self._output_fake_response('output2')
    list_text = self.shell('output-show -F json --with-detail teststack/1 output2')
    required = ['output_key', 'output_value', 'description', 'output2', '[\n    "output", \n    "value", \n    "2"\n  ]test output 2']
    for r in required:
        self.assertRegex(list_text, r)