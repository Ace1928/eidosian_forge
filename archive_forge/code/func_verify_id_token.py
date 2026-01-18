import collections
import copy
import datetime
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import six
from six.moves import http_client
from six.moves import urllib
import oauth2client
from oauth2client import _helpers
from oauth2client import _pkce
from oauth2client import clientsecrets
from oauth2client import transport
@_helpers.positional(2)
def verify_id_token(id_token, audience, http=None, cert_uri=ID_TOKEN_VERIFICATION_CERTS):
    """Verifies a signed JWT id_token.

    This function requires PyOpenSSL and because of that it does not work on
    App Engine.

    Args:
        id_token: string, A Signed JWT.
        audience: string, The audience 'aud' that the token should be for.
        http: httplib2.Http, instance to use to make the HTTP request. Callers
              should supply an instance that has caching enabled.
        cert_uri: string, URI of the certificates in JSON format to
                  verify the JWT against.

    Returns:
        The deserialized JSON in the JWT.

    Raises:
        oauth2client.crypt.AppIdentityError: if the JWT fails to verify.
        CryptoUnavailableError: if no crypto library is available.
    """
    _require_crypto_or_die()
    if http is None:
        http = transport.get_cached_http()
    resp, content = transport.request(http, cert_uri)
    if resp.status == http_client.OK:
        certs = json.loads(_helpers._from_bytes(content))
        return crypt.verify_signed_jwt_with_certs(id_token, certs, audience)
    else:
        raise VerifyJwtTokenError('Status code: {0}'.format(resp.status))