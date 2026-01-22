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
class OAuth2AuthnMethodsTests(test_v3.OAuth2RestfulTestCase):
    ACCESS_TOKEN_URL = '/OS-OAUTH2/token'

    def setUp(self):
        super(OAuth2AuthnMethodsTests, self).setUp()
        self.config_fixture.config(group='oauth2', oauth2_authn_methods=['client_secret_basic', 'tls_client_auth'])

    def _get_access_token(self, headers, data, expected_status, client_cert_content=None):
        data = parse.urlencode(data).encode()
        kwargs = {'headers': headers, 'noauth': True, 'convert': False, 'body': data, 'expected_status': expected_status}
        if client_cert_content:
            kwargs.update({'environ': {'SSL_CLIENT_CERT': client_cert_content}})
        resp = self.post(self.ACCESS_TOKEN_URL, **kwargs)
        return resp

    def _create_certificates(self):
        return unit.create_certificate(subject_dn=unit.create_dn(country_name='jp', state_or_province_name='tokyo', locality_name='musashino', organizational_unit_name='test'))

    def _get_cert_content(self, cert):
        return cert.public_bytes(Encoding.PEM).decode('ascii')

    @mock.patch.object(AccessTokenResource, '_client_secret_basic')
    def test_secret_basic_header(self, mock_client_secret_basic):
        """client_secret_basic is used if a client sercret is found."""
        client_id = 'client_id'
        client_secret = 'client_secret'
        b64str = b64encode(f'{client_id}:{client_secret}'.encode()).decode().strip()
        headers = {'Content-Type': 'application/x-www-form-urlencoded', 'Authorization': f'Basic {b64str}'}
        data = {'grant_type': 'client_credentials'}
        _ = self._get_access_token(headers=headers, data=data, expected_status=client.OK)
        mock_client_secret_basic.assert_called_once_with(client_id, client_secret)

    @mock.patch.object(AccessTokenResource, '_client_secret_basic')
    def test_secret_basic_form(self, mock_client_secret_basic):
        """client_secret_basic is used if a client sercret is found."""
        client_id = 'client_id'
        client_secret = 'client_secret'
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {'grant_type': 'client_credentials', 'client_id': client_id, 'client_secret': client_secret}
        _ = self._get_access_token(headers=headers, data=data, expected_status=client.OK)
        mock_client_secret_basic.assert_called_once_with(client_id, client_secret)

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

    @mock.patch.object(AccessTokenResource, '_tls_client_auth')
    def test_client_cert(self, mock_tls_client_auth):
        """tls_client_auth is used if a certificate is found."""
        client_id = 'client_id'
        client_cert, _ = self._create_certificates()
        cert_content = self._get_cert_content(client_cert)
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {'grant_type': 'client_credentials', 'client_id': client_id}
        _ = self._get_access_token(headers=headers, data=data, expected_status=client.OK, client_cert_content=cert_content)
        mock_tls_client_auth.assert_called_once_with(client_id, cert_content)

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