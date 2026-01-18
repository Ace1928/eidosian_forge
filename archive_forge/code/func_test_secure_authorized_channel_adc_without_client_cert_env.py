import datetime
import os
import time
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import credentials
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import service_account
@mock.patch('google.auth.transport.grpc.SslCredentials', autospec=True)
def test_secure_authorized_channel_adc_without_client_cert_env(self, ssl_credentials_adc_method, secure_channel, ssl_channel_credentials, metadata_call_credentials, composite_channel_credentials, get_client_ssl_credentials):
    credentials = CredentialsStub()
    request = mock.create_autospec(transport.Request)
    target = 'example.com:80'
    channel = google.auth.transport.grpc.secure_authorized_channel(credentials, request, target, options=mock.sentinel.options)
    auth_plugin = metadata_call_credentials.call_args[0][0]
    assert isinstance(auth_plugin, google.auth.transport.grpc.AuthMetadataPlugin)
    assert auth_plugin._credentials == credentials
    assert auth_plugin._request == request
    ssl_channel_credentials.assert_called_once()
    ssl_credentials_adc_method.assert_not_called()
    composite_channel_credentials.assert_called_once_with(ssl_channel_credentials.return_value, metadata_call_credentials.return_value)
    secure_channel.assert_called_once_with(target, composite_channel_credentials.return_value, options=mock.sentinel.options)
    assert channel == secure_channel.return_value