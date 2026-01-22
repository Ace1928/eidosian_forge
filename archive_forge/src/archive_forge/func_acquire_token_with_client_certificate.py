from __future__ import absolute_import, division, print_function
import os
import re
import types
import copy
import inspect
import traceback
import json
from os.path import expanduser
from ansible.module_utils.basic import \
from ansible.module_utils.six.moves import configparser
import ansible.module_utils.six.moves.urllib.parse as urlparse
from base64 import b64encode, b64decode
from hashlib import sha256
from hmac import HMAC
from time import time
def acquire_token_with_client_certificate(self, authority, x509_private_key_path, thumbprint, client_id, tenant):
    authority_uri = authority
    if tenant is not None:
        authority_uri = authority + '/' + tenant
    x509_private_key = None
    with open(x509_private_key_path, 'r') as pem_file:
        x509_private_key = pem_file.read()
    base_url = self._cloud_environment.endpoints.resource_manager
    if not base_url.endswith('/'):
        base_url += '/'
    scopes = [base_url + '.default']
    client_credential = {'thumbprint': thumbprint, 'private_key': x509_private_key}
    context = ConfidentialClientApplication(client_id=client_id, authority=authority_uri, client_credential=client_credential)
    token_response = context.acquire_token_for_client(scopes=scopes)
    return AADTokenCredentials(token_response)