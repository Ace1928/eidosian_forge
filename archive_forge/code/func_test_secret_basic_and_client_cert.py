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
@mock.patch.object(AccessTokenResource, '_tls_client_auth')
def test_secret_basic_and_client_cert(self, mock_tls_client_auth):
    """tls_client_auth is used if a certificate and secret are found."""
    client_id_s = 'client_id_s'
    client_secret = 'client_secret'
    client_id_c = 'client_id_c'
    client_cert, _ = self._create_certificates()
    cert_content = self._get_cert_content(client_cert)
    b64str = b64encode(f'{client_id_s}:{client_secret}'.encode()).decode().strip()
    headers = {'Content-Type': 'application/x-www-form-urlencoded', 'Authorization': f'Basic {b64str}'}
    data = {'grant_type': 'client_credentials', 'client_id': client_id_c}
    _ = self._get_access_token(headers=headers, data=data, expected_status=client.OK, client_cert_content=cert_content)
    mock_tls_client_auth.assert_called_once_with(client_id_c, cert_content)