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
@mock.patch('sys.stdin', new_callable=io.StringIO)
def test_stack_delete_prompt_with_tty_y(self, ms):
    self.register_keystone_auth_fixture()
    mock_stdin = mock.Mock()
    mock_stdin.isatty = mock.Mock()
    mock_stdin.isatty.return_value = True
    mock_stdin.readline = mock.Mock()
    mock_stdin.readline.return_value = ''
    mock_stdin.fileno.return_value = 0
    sys.stdin = mock_stdin
    self.mock_request_delete('/stacks/teststack2/2')
    resp = self.shell('stack-delete -y teststack2/2')
    msg = 'Request to delete stack teststack2/2 has been accepted.'
    self.assertRegex(resp, msg)