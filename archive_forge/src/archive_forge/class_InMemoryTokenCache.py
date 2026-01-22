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
class InMemoryTokenCache(TokenCache):
    """An in-memory token cache.

  The cache is implemented by a python dict, and inherits the thread-safety
  properties of dict.
  """

    def __init__(self):
        super(InMemoryTokenCache, self).__init__()
        self.cache = dict()

    def PutToken(self, key, value):
        LOG.debug('InMemoryTokenCache.PutToken: key=%s', key)
        self.cache[key] = value

    def GetToken(self, key):
        value = self.cache.get(key, None)
        LOG.debug('InMemoryTokenCache.GetToken: key=%s%s present', key, ' not' if value is None else '')
        return value