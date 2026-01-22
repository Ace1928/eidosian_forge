import copy
import datetime
import io
import json
from google.auth import aws
from google.auth import credentials
from google.auth import exceptions
from google.auth import external_account
from google.auth import external_account_authorized_user
from google.auth import identity_pool
from google.auth import pluggable
from google.auth.transport import requests
from gslib.utils import constants
import oauth2client
class WrappedCredentials(oauth2client.client.OAuth2Credentials):
    """A utility class to use Google Auth credentials in place of oauth2client credentials.
  """
    NON_SERIALIZED_MEMBERS = frozenset(list(oauth2client.client.OAuth2Credentials.NON_SERIALIZED_MEMBERS) + ['_base'])

    def __init__(self, base_creds):
        """Initializes oauth2client credentials based on underlying Google Auth credentials.

    Args:
      external_account_creds: subclass of google.auth.external_account.Credentials
    """
        self._base = base_creds
        if isinstance(base_creds, external_account.Credentials):
            client_id = self._base._audience
            client_secret = None
            refresh_token = None
        elif isinstance(base_creds, external_account_authorized_user.Credentials):
            client_id = self._base.client_id
            client_secret = self._base.client_secret
            refresh_token = self._base.refresh_token
        else:
            raise TypeError('Invalid Credentials')
        super(WrappedCredentials, self).__init__(access_token=self._base.token, client_id=client_id, client_secret=client_secret, refresh_token=refresh_token, token_expiry=self._base.expiry, token_uri=None, user_agent=None)

    def _do_refresh_request(self, http):
        self._base.refresh(requests.Request())
        if self.store is not None:
            self.store.locked_put(self)

    @property
    def access_token(self):
        return self._base.token

    @access_token.setter
    def access_token(self, value):
        self._base.token = value

    @property
    def token_expiry(self):
        return self._base.expiry

    @token_expiry.setter
    def token_expiry(self, value):
        self._base.expiry = value

    def to_json(self):
        """Utility function that creates JSON repr. of a Credentials object.

    Returns:
        string, a JSON representation of this instance, suitable to pass to
        from_json().
    """
        serialized_data = super().to_json()
        deserialized_data = json.loads(serialized_data)
        deserialized_data['_base'] = copy.copy(self._base.info)
        deserialized_data['access_token'] = self._base.token
        deserialized_data['token_expiry'] = _parse_expiry(self.token_expiry)
        return json.dumps(deserialized_data)

    @classmethod
    def for_external_account(cls, filename):
        creds = _get_external_account_credentials_from_file(filename)
        return cls(creds)

    @classmethod
    def for_external_account_authorized_user(cls, filename):
        creds = _get_external_account_authorized_user_credentials_from_file(filename)
        return cls(creds)

    @classmethod
    def from_json(cls, json_data):
        """Instantiate a Credentials object from a JSON description of it.

    The JSON should have been produced by calling .to_json() on the object.

    Args:
        data: dict, A deserialized JSON object.

    Returns:
        An instance of a Credentials subclass.
    """
        data = json.loads(json_data)
        base = data.get('_base')
        base_creds = None
        if base.get('type') == 'external_account':
            base_creds = _get_external_account_credentials_from_info(base)
        elif base.get('type') == 'external_account_authorized_user':
            base_creds = _get_external_account_authorized_user_credentials_from_info(base)
        creds = cls(base_creds)
        creds.access_token = data.get('access_token')
        if data.get('token_expiry') and (not isinstance(data['token_expiry'], datetime.datetime)):
            try:
                data['token_expiry'] = datetime.datetime.strptime(data['token_expiry'], oauth2client.client.EXPIRY_FORMAT)
            except ValueError:
                data['token_expiry'] = None
        creds.token_expiry = data.get('token_expiry')
        return creds