import base64
import copy
import datetime
import json
import time
import oauth2client
from oauth2client import _helpers
from oauth2client import client
from oauth2client import crypt
from oauth2client import transport
@classmethod
def _from_parsed_json_keyfile(cls, keyfile_dict, scopes, token_uri=None, revoke_uri=None):
    """Helper for factory constructors from JSON keyfile.

        Args:
            keyfile_dict: dict-like object, The parsed dictionary-like object
                          containing the contents of the JSON keyfile.
            scopes: List or string, Scopes to use when acquiring an
                    access token.
            token_uri: string, URI for OAuth 2.0 provider token endpoint.
                       If unset and not present in keyfile_dict, defaults
                       to Google's endpoints.
            revoke_uri: string, URI for OAuth 2.0 provider revoke endpoint.
                       If unset and not present in keyfile_dict, defaults
                       to Google's endpoints.

        Returns:
            ServiceAccountCredentials, a credentials object created from
            the keyfile contents.

        Raises:
            ValueError, if the credential type is not :data:`SERVICE_ACCOUNT`.
            KeyError, if one of the expected keys is not present in
                the keyfile.
        """
    creds_type = keyfile_dict.get('type')
    if creds_type != client.SERVICE_ACCOUNT:
        raise ValueError('Unexpected credentials type', creds_type, 'Expected', client.SERVICE_ACCOUNT)
    service_account_email = keyfile_dict['client_email']
    private_key_pkcs8_pem = keyfile_dict['private_key']
    private_key_id = keyfile_dict['private_key_id']
    client_id = keyfile_dict['client_id']
    if not token_uri:
        token_uri = keyfile_dict.get('token_uri', oauth2client.GOOGLE_TOKEN_URI)
    if not revoke_uri:
        revoke_uri = keyfile_dict.get('revoke_uri', oauth2client.GOOGLE_REVOKE_URI)
    signer = crypt.Signer.from_string(private_key_pkcs8_pem)
    credentials = cls(service_account_email, signer, scopes=scopes, private_key_id=private_key_id, client_id=client_id, token_uri=token_uri, revoke_uri=revoke_uri)
    credentials._private_key_pkcs8_pem = private_key_pkcs8_pem
    return credentials