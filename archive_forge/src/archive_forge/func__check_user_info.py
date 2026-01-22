import json
import os
from six.moves import http_client
import oauth2client
from oauth2client import client
from oauth2client import service_account
from oauth2client import transport
def _check_user_info(credentials, expected_email):
    http = credentials.authorize(transport.get_http_object())
    response, content = transport.request(http, USER_INFO)
    if response.status != http_client.OK:
        raise ValueError('Expected 200 OK response.')
    content = content.decode('utf-8')
    payload = json.loads(content)
    if payload['email'] != expected_email:
        raise ValueError('User info email does not match credentials.')