import os
import sys
import time
import errno
import base64
import logging
import datetime
import urllib.parse
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from libcloud.utils.py3 import b, httplib, urlparse, urlencode
from libcloud.common.base import BaseDriver, JsonResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, ProviderError
from libcloud.utils.connection import get_response_object
class GoogleGCEServiceAcctAuthConnection(GoogleBaseAuthConnection):
    """Authentication class for self-authentication when used with a GCE
    instance that supports serviceAccounts.
    """

    def get_new_token(self):
        """
        Get a new token from the internal metadata service.

        :return:  Dictionary containing token information
        :rtype:   ``dict``
        """
        path = '/instance/service-accounts/default/token'
        http_code, http_reason, token_info = _get_gce_metadata(path)
        if http_code == httplib.NOT_FOUND:
            raise ValueError('Service Accounts are not enabled for this GCE instance.')
        if http_code != httplib.OK:
            raise ValueError("Internal GCE Authorization failed: '%s'" % str(http_reason))
        token_info = json.loads(token_info)
        if 'expires_in' in token_info:
            expire_time = _utcnow() + datetime.timedelta(seconds=token_info['expires_in'])
            token_info['expire_time'] = _utc_timestamp(expire_time)
        return token_info