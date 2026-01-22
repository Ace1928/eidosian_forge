import json
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import utils
class TestOAuthClientAuthHandler(object):
    CLIENT_AUTH_BASIC = utils.ClientAuthentication(utils.ClientAuthType.basic, CLIENT_ID, CLIENT_SECRET)
    CLIENT_AUTH_BASIC_SECRETLESS = utils.ClientAuthentication(utils.ClientAuthType.basic, CLIENT_ID)
    CLIENT_AUTH_REQUEST_BODY = utils.ClientAuthentication(utils.ClientAuthType.request_body, CLIENT_ID, CLIENT_SECRET)
    CLIENT_AUTH_REQUEST_BODY_SECRETLESS = utils.ClientAuthentication(utils.ClientAuthType.request_body, CLIENT_ID)

    @classmethod
    def make_oauth_client_auth_handler(cls, client_auth=None):
        return AuthHandler(client_auth)

    def test_apply_client_authentication_options_none(self):
        headers = {'Content-Type': 'application/json'}
        request_body = {'foo': 'bar'}
        auth_handler = self.make_oauth_client_auth_handler()
        auth_handler.apply_client_authentication_options(headers, request_body)
        assert headers == {'Content-Type': 'application/json'}
        assert request_body == {'foo': 'bar'}

    def test_apply_client_authentication_options_basic(self):
        headers = {'Content-Type': 'application/json'}
        request_body = {'foo': 'bar'}
        auth_handler = self.make_oauth_client_auth_handler(self.CLIENT_AUTH_BASIC)
        auth_handler.apply_client_authentication_options(headers, request_body)
        assert headers == {'Content-Type': 'application/json', 'Authorization': 'Basic {}'.format(BASIC_AUTH_ENCODING)}
        assert request_body == {'foo': 'bar'}

    def test_apply_client_authentication_options_basic_nosecret(self):
        headers = {'Content-Type': 'application/json'}
        request_body = {'foo': 'bar'}
        auth_handler = self.make_oauth_client_auth_handler(self.CLIENT_AUTH_BASIC_SECRETLESS)
        auth_handler.apply_client_authentication_options(headers, request_body)
        assert headers == {'Content-Type': 'application/json', 'Authorization': 'Basic {}'.format(BASIC_AUTH_ENCODING_SECRETLESS)}
        assert request_body == {'foo': 'bar'}

    def test_apply_client_authentication_options_request_body(self):
        headers = {'Content-Type': 'application/json'}
        request_body = {'foo': 'bar'}
        auth_handler = self.make_oauth_client_auth_handler(self.CLIENT_AUTH_REQUEST_BODY)
        auth_handler.apply_client_authentication_options(headers, request_body)
        assert headers == {'Content-Type': 'application/json'}
        assert request_body == {'foo': 'bar', 'client_id': CLIENT_ID, 'client_secret': CLIENT_SECRET}

    def test_apply_client_authentication_options_request_body_nosecret(self):
        headers = {'Content-Type': 'application/json'}
        request_body = {'foo': 'bar'}
        auth_handler = self.make_oauth_client_auth_handler(self.CLIENT_AUTH_REQUEST_BODY_SECRETLESS)
        auth_handler.apply_client_authentication_options(headers, request_body)
        assert headers == {'Content-Type': 'application/json'}
        assert request_body == {'foo': 'bar', 'client_id': CLIENT_ID, 'client_secret': ''}

    def test_apply_client_authentication_options_request_body_no_body(self):
        headers = {'Content-Type': 'application/json'}
        auth_handler = self.make_oauth_client_auth_handler(self.CLIENT_AUTH_REQUEST_BODY)
        with pytest.raises(exceptions.OAuthError) as excinfo:
            auth_handler.apply_client_authentication_options(headers)
        assert excinfo.match('HTTP request does not support request-body')

    def test_apply_client_authentication_options_bearer_token(self):
        bearer_token = 'ACCESS_TOKEN'
        headers = {'Content-Type': 'application/json'}
        request_body = {'foo': 'bar'}
        auth_handler = self.make_oauth_client_auth_handler()
        auth_handler.apply_client_authentication_options(headers, request_body, bearer_token)
        assert headers == {'Content-Type': 'application/json', 'Authorization': 'Bearer {}'.format(bearer_token)}
        assert request_body == {'foo': 'bar'}

    def test_apply_client_authentication_options_bearer_and_basic(self):
        bearer_token = 'ACCESS_TOKEN'
        headers = {'Content-Type': 'application/json'}
        request_body = {'foo': 'bar'}
        auth_handler = self.make_oauth_client_auth_handler(self.CLIENT_AUTH_BASIC)
        auth_handler.apply_client_authentication_options(headers, request_body, bearer_token)
        assert headers == {'Content-Type': 'application/json', 'Authorization': 'Bearer {}'.format(bearer_token)}
        assert request_body == {'foo': 'bar'}

    def test_apply_client_authentication_options_bearer_and_request_body(self):
        bearer_token = 'ACCESS_TOKEN'
        headers = {'Content-Type': 'application/json'}
        request_body = {'foo': 'bar'}
        auth_handler = self.make_oauth_client_auth_handler(self.CLIENT_AUTH_REQUEST_BODY)
        auth_handler.apply_client_authentication_options(headers, request_body, bearer_token)
        assert headers == {'Content-Type': 'application/json', 'Authorization': 'Bearer {}'.format(bearer_token)}
        assert request_body == {'foo': 'bar'}