from base64 import b64encode
from cryptography.hazmat.primitives.serialization import Encoding
import fixtures
import http
from http import client
from oslo_log import log
from oslo_serialization import jsonutils
from unittest import mock
from urllib import parse
from keystone.api.os_oauth2 import AccessTokenResource
from keystone.common import provider_api
from keystone.common import utils
from keystone import conf
from keystone import exception
from keystone.federation.utils import RuleProcessor
from keystone.tests import unit
from keystone.tests.unit import test_v3
from keystone.token.provider import Manager
@mock.patch.object(AccessTokenResource, '_client_secret_basic')
def test_secret_basic_header_and_form(self, mock_client_secret_basic):
    """A header is used if secrets are found in a header and body."""
    client_id_h = 'client_id_h'
    client_secret_h = 'client_secret_h'
    client_id_d = 'client_id_d'
    client_secret_d = 'client_secret_d'
    b64str = b64encode(f'{client_id_h}:{client_secret_h}'.encode()).decode().strip()
    headers = {'Content-Type': 'application/x-www-form-urlencoded', 'Authorization': f'Basic {b64str}'}
    data = {'grant_type': 'client_credentials', 'client_id': client_id_d, 'client_secret': client_secret_d}
    _ = self._get_access_token(headers=headers, data=data, expected_status=client.OK)
    mock_client_secret_basic.assert_called_once_with(client_id_h, client_secret_h)