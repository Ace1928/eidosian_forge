import re
from _pytest.capture import CaptureFixture
import authenticate_explicit_with_adc
import authenticate_implicit_with_adc
import idtoken_from_metadata_server
import idtoken_from_service_account
import verify_google_idtoken
import google
from google.oauth2 import service_account
import google.auth.transport.requests
import os
def test_verify_google_idtoken():
    idtoken = get_idtoken_from_service_account(SERVICE_ACCOUNT_FILE, 'iap.googleapis.com')
    verify_google_idtoken.verify_google_idtoken(idtoken, 'iap.googleapis.com', 'https://www.googleapis.com/oauth2/v3/certs')