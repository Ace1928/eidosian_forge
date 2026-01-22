import json
import os
from six.moves import http_client
import oauth2client
from oauth2client import client
from oauth2client import service_account
from oauth2client import transport
def _require_environ():
    if JSON_KEY_PATH is None or P12_KEY_PATH is None or P12_KEY_EMAIL is None or (USER_KEY_PATH is None) or (USER_KEY_EMAIL is None):
        raise EnvironmentError('Expected environment variables to be set:', 'OAUTH2CLIENT_TEST_JSON_KEY_PATH', 'OAUTH2CLIENT_TEST_P12_KEY_PATH', 'OAUTH2CLIENT_TEST_P12_KEY_EMAIL', 'OAUTH2CLIENT_TEST_USER_KEY_PATH', 'OAUTH2CLIENT_TEST_USER_KEY_EMAIL')
    if not os.path.isfile(JSON_KEY_PATH):
        raise EnvironmentError(JSON_KEY_PATH, 'is not a file')
    if not os.path.isfile(P12_KEY_PATH):
        raise EnvironmentError(P12_KEY_PATH, 'is not a file')
    if not os.path.isfile(USER_KEY_PATH):
        raise EnvironmentError(USER_KEY_PATH, 'is not a file')