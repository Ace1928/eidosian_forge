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
def _do_revoke(self, http, token):
    """Revokes this credential and deletes the stored copy (if it exists).

        Args:
            http: an object to be used to make HTTP requests.
            token: A string used as the token to be revoked. Can be either an
                   access_token or refresh_token.

        Raises:
            TokenRevokeError: If the revoke request does not return with a
                              200 OK.
        """
    logger.info('Revoking token')
    query_params = {'token': token}
    token_revoke_uri = _helpers.update_query_params(self.revoke_uri, query_params)
    resp, content = transport.request(http, token_revoke_uri)
    if resp.status == http_client.METHOD_NOT_ALLOWED:
        body = urllib.parse.urlencode(query_params)
        resp, content = transport.request(http, token_revoke_uri, method='POST', body=body)
    if resp.status == http_client.OK:
        self.invalid = True
    else:
        error_msg = 'Invalid response {0}.'.format(resp.status)
        try:
            d = json.loads(_helpers._from_bytes(content))
            if 'error' in d:
                error_msg = d['error']
        except (TypeError, ValueError):
            pass
        raise TokenRevokeError(error_msg)
    if self.store:
        self.store.delete()