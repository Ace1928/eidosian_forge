import datetime
import errno
import json
import os
import requests
import sys
import time
import webbrowser
import google_auth_oauthlib.flow as auth_flows
import grpc
import google.auth
import google.auth.transport.requests
import google.oauth2.credentials
from tensorboard.uploader import util
from tensorboard.util import tb_logging
def write_credentials(self, credentials):
    """Writes a `google.oauth2.credentials.Credentials` to the store."""
    if not isinstance(credentials, google.oauth2.credentials.Credentials):
        raise TypeError('Cannot write credentials of type %s' % type(credentials))
    if self._credentials_filepath is None:
        return
    private = os.name != 'nt'
    util.make_file_with_directories(self._credentials_filepath, private=private)
    data = {'refresh_token': credentials.refresh_token, 'token_uri': credentials.token_uri, 'client_id': credentials.client_id, 'client_secret': credentials.client_secret, 'scopes': credentials.scopes, 'type': 'authorized_user'}
    with open(self._credentials_filepath, 'w') as f:
        json.dump(data, f)