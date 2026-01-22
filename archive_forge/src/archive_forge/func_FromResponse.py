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
@classmethod
def FromResponse(cls, response):
    """Create a DeviceFlowInfo from a server response.

        The response should be a dict containing entries as described here:

        http://tools.ietf.org/html/draft-ietf-oauth-v2-05#section-3.7.1
        """
    kwargs = {'device_code': response['device_code'], 'user_code': response['user_code']}
    verification_url = response.get('verification_url', response.get('verification_uri'))
    if verification_url is None:
        raise OAuth2DeviceCodeError('No verification_url provided in server response')
    kwargs['verification_url'] = verification_url
    kwargs.update({'interval': response.get('interval'), 'user_code_expiry': None})
    if 'expires_in' in response:
        kwargs['user_code_expiry'] = _UTCNOW() + datetime.timedelta(seconds=int(response['expires_in']))
    return cls(**kwargs)