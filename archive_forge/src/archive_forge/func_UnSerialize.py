from __future__ import absolute_import
import datetime
import errno
from hashlib import sha1
import json
import logging
import os
import socket
import tempfile
import threading
import boto
import httplib2
import oauth2client.client
import oauth2client.service_account
from google_reauth import reauth_creds
import retry_decorator.retry_decorator
import six
from six import BytesIO
from six.moves import urllib
@staticmethod
def UnSerialize(query):
    """Creates an AccessToken object from its serialized form."""

    def GetValue(d, key):
        return d.get(key, [None])[0]
    kv = urllib.parse.parse_qs(query)
    if 'token' not in kv or not kv['token']:
        return None
    expiry = None
    expiry_tuple = GetValue(kv, 'expiry')
    if expiry_tuple:
        try:
            expiry = datetime.datetime(*[int(n) for n in expiry_tuple.split(',')])
        except:
            return None
    rapt_token = GetValue(kv, 'rapt_token')
    return AccessToken(GetValue(kv, 'token'), expiry, rapt_token=rapt_token)