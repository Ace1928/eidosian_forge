from http import client as http_client
import json
import time
from unittest import mock
from keystoneauth1 import exceptions as kexc
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient import exc
from ironicclient.tests.unit import utils
def test_server_exception_description_only(self):
    error_msg = 'test error msg'
    error_body = _get_error_body(description=error_msg)
    fake_session = utils.mockSession({'Content-Type': 'application/json'}, error_body, status_code=http_client.BAD_REQUEST)
    client = _session_client(session=fake_session)
    self.assertRaisesRegex(exc.BadRequest, 'test error msg', client.json_request, 'GET', '/v1/resources')