import copy
import datetime
import json
import os
import mock
import pytest  # type: ignore
import requests
import six
from google.auth import exceptions
from google.auth import jwt
import google.auth.transport.requests
from google.oauth2 import gdch_credentials
from google.oauth2.gdch_credentials import ServiceAccountCredentials
def test_with_gdch_audience(self):
    mock_signer = mock.Mock()
    creds = ServiceAccountCredentials._from_signer_and_info(mock_signer, self.INFO)
    assert creds._signer == mock_signer
    assert creds._service_identity_name == self.NAME
    assert creds._audience is None
    assert creds._token_uri == self.TOKEN_URI
    assert creds._ca_cert_path == self.CA_CERT_PATH
    new_creds = creds.with_gdch_audience(self.AUDIENCE)
    assert new_creds._signer == mock_signer
    assert new_creds._service_identity_name == self.NAME
    assert new_creds._audience == self.AUDIENCE
    assert new_creds._token_uri == self.TOKEN_URI
    assert new_creds._ca_cert_path == self.CA_CERT_PATH