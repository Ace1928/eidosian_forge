import json
import pytest
import google.oauth2.credentials
from google.oauth2 import service_account
import google.auth.impersonated_credentials
from google.auth import _helpers
def test_refresh_with_service_account_credentials_as_source(http_request, service_account_credentials, impersonated_service_account_credentials, token_info):
    source_credentials = service_account_credentials.with_scopes(['email'])
    source_credentials.refresh(http_request)
    assert source_credentials.token
    target_scopes = ['https://www.googleapis.com/auth/devstorage.read_only', 'https://www.googleapis.com/auth/analytics']
    target_credentials = google.auth.impersonated_credentials.Credentials(source_credentials=source_credentials, target_principal=impersonated_service_account_credentials.service_account_email, target_scopes=target_scopes)
    target_credentials.refresh(http_request)
    assert target_credentials.token