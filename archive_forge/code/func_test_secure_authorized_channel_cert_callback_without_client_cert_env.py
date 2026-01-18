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
def test_secure_authorized_channel_cert_callback_without_client_cert_env(self, secure_channel, ssl_channel_credentials, metadata_call_credentials, composite_channel_credentials, get_client_ssl_credentials):
    credentials = mock.Mock()
    request = mock.Mock()
    target = 'example.com:80'
    client_cert_callback = mock.Mock()
    google.auth.transport.grpc.secure_authorized_channel(credentials, request, target, client_cert_callback=client_cert_callback)
    client_cert_callback.assert_not_called()
    ssl_channel_credentials.assert_called_once()
    composite_channel_credentials.assert_called_once_with(ssl_channel_credentials.return_value, metadata_call_credentials.return_value)