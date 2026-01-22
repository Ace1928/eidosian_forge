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
class CredentialsStore:
    """Private file store for a `google.oauth2.credentials.Credentials`."""
    _DEFAULT_CONFIG_DIRECTORY = object()

    def __init__(self, user_config_directory=_DEFAULT_CONFIG_DIRECTORY):
        """Creates a CredentialsStore.

        Args:
          user_config_directory: Optional absolute path to the root directory for
            storing user configs, under which to store the credentials file. If not
            set, defaults to a platform-specific location. If set to None, the
            store is disabled (reads return None; write and clear are no-ops).
        """
        if user_config_directory is CredentialsStore._DEFAULT_CONFIG_DIRECTORY:
            user_config_directory = util.get_user_config_directory()
            if user_config_directory is None:
                logger.warning('Credentials caching disabled - no private config directory found')
        if user_config_directory is None:
            self._credentials_filepath = None
        else:
            self._credentials_filepath = os.path.join(user_config_directory, *TENSORBOARD_CREDENTIALS_FILEPATH_PARTS)

    def read_credentials(self):
        """Returns the current `google.oauth2.credentials.Credentials`, or
        None."""
        if self._credentials_filepath is None:
            return None
        if os.path.exists(self._credentials_filepath):
            return google.oauth2.credentials.Credentials.from_authorized_user_file(self._credentials_filepath)
        return None

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

    def clear(self):
        """Clears the store of any persisted credentials information."""
        if self._credentials_filepath is None:
            return
        try:
            os.remove(self._credentials_filepath)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise