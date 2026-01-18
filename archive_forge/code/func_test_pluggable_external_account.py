import datetime
import json
import os
import socket
from tempfile import NamedTemporaryFile
import threading
import time
import sys
import google.auth
from google.auth import _helpers
from googleapiclient import discovery
from six.moves import BaseHTTPServer
from google.oauth2 import service_account
import pytest
from mock import patch
def test_pluggable_external_account(oidc_credentials, service_account_info, dns_access):
    now = datetime.datetime.now()
    unix_seconds = time.mktime(now.timetuple())
    expiration_time = (unix_seconds + 1 * 60 * 60) * 1000
    credential = {'success': True, 'version': 1, 'expiration_time': expiration_time, 'token_type': 'urn:ietf:params:oauth:token-type:jwt', 'id_token': oidc_credentials.token}
    tmpfile = NamedTemporaryFile(delete=True)
    with open(tmpfile.name, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('echo "{}"\n'.format(json.dumps(credential).replace('"', '\\"')))
    tmpfile.file.close()
    os.chmod(tmpfile.name, 511)
    assert get_project_dns(dns_access, {'type': 'external_account', 'audience': _AUDIENCE_OIDC, 'subject_token_type': 'urn:ietf:params:oauth:token-type:jwt', 'token_url': 'https://sts.googleapis.com/v1/token', 'service_account_impersonation_url': 'https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{}:generateAccessToken'.format(oidc_credentials.service_account_email), 'credential_source': {'executable': {'command': tmpfile.name}}})