import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from google.auth import _helpers
from google.auth import crypt
from google.auth import exceptions
from google.auth import impersonated_credentials
from google.auth import transport
from google.auth.impersonated_credentials import Credentials
from google.oauth2 import credentials
from google.oauth2 import service_account
class TestImpersonatedCredentials(object):
    SERVICE_ACCOUNT_EMAIL = 'service-account@example.com'
    TARGET_PRINCIPAL = 'impersonated@project.iam.gserviceaccount.com'
    TARGET_SCOPES = ['https://www.googleapis.com/auth/devstorage.read_only']
    DELEGATES = []
    LIFETIME = 3600
    SOURCE_CREDENTIALS = service_account.Credentials(SIGNER, SERVICE_ACCOUNT_EMAIL, TOKEN_URI)
    USER_SOURCE_CREDENTIALS = credentials.Credentials(token='ABCDE')
    IAM_ENDPOINT_OVERRIDE = 'https://us-east1-iamcredentials.googleapis.com/v1/projects/-' + '/serviceAccounts/{}:generateAccessToken'.format(SERVICE_ACCOUNT_EMAIL)

    def make_credentials(self, source_credentials=SOURCE_CREDENTIALS, lifetime=LIFETIME, target_principal=TARGET_PRINCIPAL, iam_endpoint_override=None):
        return Credentials(source_credentials=source_credentials, target_principal=target_principal, target_scopes=self.TARGET_SCOPES, delegates=self.DELEGATES, lifetime=lifetime, iam_endpoint_override=iam_endpoint_override)

    def test_make_from_user_credentials(self):
        credentials = self.make_credentials(source_credentials=self.USER_SOURCE_CREDENTIALS)
        assert not credentials.valid
        assert credentials.expired

    def test_default_state(self):
        credentials = self.make_credentials()
        assert not credentials.valid
        assert credentials.expired

    def make_request(self, data, status=http_client.OK, headers=None, side_effect=None, use_data_bytes=True):
        response = mock.create_autospec(transport.Response, instance=False)
        response.status = status
        response.data = _helpers.to_bytes(data) if use_data_bytes else data
        response.headers = headers or {}
        request = mock.create_autospec(transport.Request, instance=False)
        request.side_effect = side_effect
        request.return_value = response
        return request

    @pytest.mark.parametrize('use_data_bytes', [True, False])
    def test_refresh_success(self, use_data_bytes, mock_donor_credentials):
        credentials = self.make_credentials(lifetime=None)
        token = 'token'
        expire_time = (_helpers.utcnow().replace(microsecond=0) + datetime.timedelta(seconds=500)).isoformat('T') + 'Z'
        response_body = {'accessToken': token, 'expireTime': expire_time}
        request = self.make_request(data=json.dumps(response_body), status=http_client.OK, use_data_bytes=use_data_bytes)
        credentials.refresh(request)
        assert credentials.valid
        assert not credentials.expired

    @pytest.mark.parametrize('use_data_bytes', [True, False])
    def test_refresh_success_iam_endpoint_override(self, use_data_bytes, mock_donor_credentials):
        credentials = self.make_credentials(lifetime=None, iam_endpoint_override=self.IAM_ENDPOINT_OVERRIDE)
        token = 'token'
        expire_time = (_helpers.utcnow().replace(microsecond=0) + datetime.timedelta(seconds=500)).isoformat('T') + 'Z'
        response_body = {'accessToken': token, 'expireTime': expire_time}
        request = self.make_request(data=json.dumps(response_body), status=http_client.OK, use_data_bytes=use_data_bytes)
        credentials.refresh(request)
        assert credentials.valid
        assert not credentials.expired
        request_kwargs = request.call_args[1]
        assert request_kwargs['url'] == self.IAM_ENDPOINT_OVERRIDE

    @pytest.mark.parametrize('time_skew', [100, -100])
    def test_refresh_source_credentials(self, time_skew):
        credentials = self.make_credentials(lifetime=None)
        credentials._source_credentials.expiry = _helpers.utcnow() + _helpers.REFRESH_THRESHOLD + datetime.timedelta(seconds=time_skew)
        credentials._source_credentials.token = 'Token'
        with mock.patch('google.oauth2.service_account.Credentials.refresh', autospec=True) as source_cred_refresh:
            expire_time = (_helpers.utcnow().replace(microsecond=0) + datetime.timedelta(seconds=500)).isoformat('T') + 'Z'
            response_body = {'accessToken': 'token', 'expireTime': expire_time}
            request = self.make_request(data=json.dumps(response_body), status=http_client.OK)
            credentials.refresh(request)
            assert credentials.valid
            assert not credentials.expired
            if time_skew > 0:
                source_cred_refresh.assert_not_called()
            else:
                source_cred_refresh.assert_called_once()

    def test_refresh_failure_malformed_expire_time(self, mock_donor_credentials):
        credentials = self.make_credentials(lifetime=None)
        token = 'token'
        expire_time = (_helpers.utcnow() + datetime.timedelta(seconds=500)).isoformat('T')
        response_body = {'accessToken': token, 'expireTime': expire_time}
        request = self.make_request(data=json.dumps(response_body), status=http_client.OK)
        with pytest.raises(exceptions.RefreshError) as excinfo:
            credentials.refresh(request)
        assert excinfo.match(impersonated_credentials._REFRESH_ERROR)
        assert not credentials.valid
        assert credentials.expired

    def test_refresh_failure_unauthorzed(self, mock_donor_credentials):
        credentials = self.make_credentials(lifetime=None)
        response_body = {'error': {'code': 403, 'message': 'The caller does not have permission', 'status': 'PERMISSION_DENIED'}}
        request = self.make_request(data=json.dumps(response_body), status=http_client.UNAUTHORIZED)
        with pytest.raises(exceptions.RefreshError) as excinfo:
            credentials.refresh(request)
        assert excinfo.match(impersonated_credentials._REFRESH_ERROR)
        assert not credentials.valid
        assert credentials.expired

    def test_refresh_failure_http_error(self, mock_donor_credentials):
        credentials = self.make_credentials(lifetime=None)
        response_body = {}
        request = self.make_request(data=json.dumps(response_body), status=http_client.HTTPException)
        with pytest.raises(exceptions.RefreshError) as excinfo:
            credentials.refresh(request)
        assert excinfo.match(impersonated_credentials._REFRESH_ERROR)
        assert not credentials.valid
        assert credentials.expired

    def test_expired(self):
        credentials = self.make_credentials(lifetime=None)
        assert credentials.expired

    def test_signer(self):
        credentials = self.make_credentials()
        assert isinstance(credentials.signer, impersonated_credentials.Credentials)

    def test_signer_email(self):
        credentials = self.make_credentials(target_principal=self.TARGET_PRINCIPAL)
        assert credentials.signer_email == self.TARGET_PRINCIPAL

    def test_service_account_email(self):
        credentials = self.make_credentials(target_principal=self.TARGET_PRINCIPAL)
        assert credentials.service_account_email == self.TARGET_PRINCIPAL

    def test_sign_bytes(self, mock_donor_credentials, mock_authorizedsession_sign):
        credentials = self.make_credentials(lifetime=None)
        token = 'token'
        expire_time = (_helpers.utcnow().replace(microsecond=0) + datetime.timedelta(seconds=500)).isoformat('T') + 'Z'
        token_response_body = {'accessToken': token, 'expireTime': expire_time}
        response = mock.create_autospec(transport.Response, instance=False)
        response.status = http_client.OK
        response.data = _helpers.to_bytes(json.dumps(token_response_body))
        request = mock.create_autospec(transport.Request, instance=False)
        request.return_value = response
        credentials.refresh(request)
        assert credentials.valid
        assert not credentials.expired
        signature = credentials.sign_bytes(b'signed bytes')
        assert signature == b'signature'

    def test_sign_bytes_failure(self):
        credentials = self.make_credentials(lifetime=None)
        with mock.patch('google.auth.transport.requests.AuthorizedSession.request', autospec=True) as auth_session:
            data = {'error': {'code': 403, 'message': 'unauthorized'}}
            auth_session.return_value = MockResponse(data, http_client.FORBIDDEN)
            with pytest.raises(exceptions.TransportError) as excinfo:
                credentials.sign_bytes(b'foo')
            assert excinfo.match("'code': 403")

    def test_with_quota_project(self):
        credentials = self.make_credentials()
        quota_project_creds = credentials.with_quota_project('project-foo')
        assert quota_project_creds._quota_project_id == 'project-foo'

    @pytest.mark.parametrize('use_data_bytes', [True, False])
    def test_with_quota_project_iam_endpoint_override(self, use_data_bytes, mock_donor_credentials):
        credentials = self.make_credentials(lifetime=None, iam_endpoint_override=self.IAM_ENDPOINT_OVERRIDE)
        token = 'token'
        quota_project_creds = credentials.with_quota_project('project-foo')
        expire_time = (_helpers.utcnow().replace(microsecond=0) + datetime.timedelta(seconds=500)).isoformat('T') + 'Z'
        response_body = {'accessToken': token, 'expireTime': expire_time}
        request = self.make_request(data=json.dumps(response_body), status=http_client.OK, use_data_bytes=use_data_bytes)
        quota_project_creds.refresh(request)
        assert quota_project_creds.valid
        assert not quota_project_creds.expired
        request_kwargs = request.call_args[1]
        assert request_kwargs['url'] == self.IAM_ENDPOINT_OVERRIDE

    def test_with_scopes(self):
        credentials = self.make_credentials()
        credentials._target_scopes = []
        assert credentials.requires_scopes is True
        credentials = credentials.with_scopes(['fake_scope1', 'fake_scope2'])
        assert credentials.requires_scopes is False
        assert credentials._target_scopes == ['fake_scope1', 'fake_scope2']

    def test_with_scopes_provide_default_scopes(self):
        credentials = self.make_credentials()
        credentials._target_scopes = []
        credentials = credentials.with_scopes(['fake_scope1'], default_scopes=['fake_scope2'])
        assert credentials._target_scopes == ['fake_scope1']

    def test_id_token_success(self, mock_donor_credentials, mock_authorizedsession_idtoken):
        credentials = self.make_credentials(lifetime=None)
        token = 'token'
        target_audience = 'https://foo.bar'
        expire_time = (_helpers.utcnow().replace(microsecond=0) + datetime.timedelta(seconds=500)).isoformat('T') + 'Z'
        response_body = {'accessToken': token, 'expireTime': expire_time}
        request = self.make_request(data=json.dumps(response_body), status=http_client.OK)
        credentials.refresh(request)
        assert credentials.valid
        assert not credentials.expired
        id_creds = impersonated_credentials.IDTokenCredentials(credentials, target_audience=target_audience)
        id_creds.refresh(request)
        assert id_creds.token == ID_TOKEN_DATA
        assert id_creds.expiry == datetime.datetime.fromtimestamp(ID_TOKEN_EXPIRY)

    def test_id_token_from_credential(self, mock_donor_credentials, mock_authorizedsession_idtoken):
        credentials = self.make_credentials(lifetime=None)
        token = 'token'
        target_audience = 'https://foo.bar'
        expire_time = (_helpers.utcnow().replace(microsecond=0) + datetime.timedelta(seconds=500)).isoformat('T') + 'Z'
        response_body = {'accessToken': token, 'expireTime': expire_time}
        request = self.make_request(data=json.dumps(response_body), status=http_client.OK)
        credentials.refresh(request)
        assert credentials.valid
        assert not credentials.expired
        new_credentials = self.make_credentials(lifetime=None)
        id_creds = impersonated_credentials.IDTokenCredentials(credentials, target_audience=target_audience, include_email=True)
        id_creds = id_creds.from_credentials(target_credentials=new_credentials)
        id_creds.refresh(request)
        assert id_creds.token == ID_TOKEN_DATA
        assert id_creds._include_email is True
        assert id_creds._target_credentials is new_credentials

    def test_id_token_with_target_audience(self, mock_donor_credentials, mock_authorizedsession_idtoken):
        credentials = self.make_credentials(lifetime=None)
        token = 'token'
        target_audience = 'https://foo.bar'
        expire_time = (_helpers.utcnow().replace(microsecond=0) + datetime.timedelta(seconds=500)).isoformat('T') + 'Z'
        response_body = {'accessToken': token, 'expireTime': expire_time}
        request = self.make_request(data=json.dumps(response_body), status=http_client.OK)
        credentials.refresh(request)
        assert credentials.valid
        assert not credentials.expired
        id_creds = impersonated_credentials.IDTokenCredentials(credentials, include_email=True)
        id_creds = id_creds.with_target_audience(target_audience=target_audience)
        id_creds.refresh(request)
        assert id_creds.token == ID_TOKEN_DATA
        assert id_creds.expiry == datetime.datetime.fromtimestamp(ID_TOKEN_EXPIRY)
        assert id_creds._include_email is True

    def test_id_token_invalid_cred(self, mock_donor_credentials, mock_authorizedsession_idtoken):
        credentials = None
        with pytest.raises(exceptions.GoogleAuthError) as excinfo:
            impersonated_credentials.IDTokenCredentials(credentials)
        assert excinfo.match('Provided Credential must be impersonated_credentials')

    def test_id_token_with_include_email(self, mock_donor_credentials, mock_authorizedsession_idtoken):
        credentials = self.make_credentials(lifetime=None)
        token = 'token'
        target_audience = 'https://foo.bar'
        expire_time = (_helpers.utcnow().replace(microsecond=0) + datetime.timedelta(seconds=500)).isoformat('T') + 'Z'
        response_body = {'accessToken': token, 'expireTime': expire_time}
        request = self.make_request(data=json.dumps(response_body), status=http_client.OK)
        credentials.refresh(request)
        assert credentials.valid
        assert not credentials.expired
        id_creds = impersonated_credentials.IDTokenCredentials(credentials, target_audience=target_audience)
        id_creds = id_creds.with_include_email(True)
        id_creds.refresh(request)
        assert id_creds.token == ID_TOKEN_DATA

    def test_id_token_with_quota_project(self, mock_donor_credentials, mock_authorizedsession_idtoken):
        credentials = self.make_credentials(lifetime=None)
        token = 'token'
        target_audience = 'https://foo.bar'
        expire_time = (_helpers.utcnow().replace(microsecond=0) + datetime.timedelta(seconds=500)).isoformat('T') + 'Z'
        response_body = {'accessToken': token, 'expireTime': expire_time}
        request = self.make_request(data=json.dumps(response_body), status=http_client.OK)
        credentials.refresh(request)
        assert credentials.valid
        assert not credentials.expired
        id_creds = impersonated_credentials.IDTokenCredentials(credentials, target_audience=target_audience)
        id_creds = id_creds.with_quota_project('project-foo')
        id_creds.refresh(request)
        assert id_creds.quota_project_id == 'project-foo'