import base64
import boto
import boto.auth_handler
import boto.exception
import boto.plugin
import boto.utils
import copy
import datetime
from email.utils import formatdate
import hmac
import os
import posixpath
from boto.compat import urllib, encodebytes, parse_qs_safe, urlparse, six
from boto.auth_handler import AuthHandler
from boto.exception import BotoClientError
from boto.utils import get_utf8able_str
class HmacKeys(object):
    """Key based Auth handler helper."""

    def __init__(self, host, config, provider, anon=False):
        if provider.access_key is None or provider.secret_key is None:
            if not anon:
                raise boto.auth_handler.NotReadyToAuthenticate()
            else:
                self._hmac = None
                self._hmac_256 = None
        self.host = host
        self.update_provider(provider)

    def update_provider(self, provider):
        self._provider = provider
        if self._provider.secret_key:
            self._hmac = hmac.new(self._provider.secret_key.encode('utf-8'), digestmod=sha)
            if sha256:
                self._hmac_256 = hmac.new(self._provider.secret_key.encode('utf-8'), digestmod=sha256)
            else:
                self._hmac_256 = None

    def algorithm(self):
        if self._hmac_256:
            return 'HmacSHA256'
        else:
            return 'HmacSHA1'

    def _get_hmac(self):
        if self._hmac_256:
            digestmod = sha256
        else:
            digestmod = sha
        return hmac.new(self._provider.secret_key.encode('utf-8'), digestmod=digestmod)

    def sign_string(self, string_to_sign):
        new_hmac = self._get_hmac()
        new_hmac.update(string_to_sign.encode('utf-8'))
        return encodebytes(new_hmac.digest()).decode('utf-8').strip()

    def __getstate__(self):
        pickled_dict = copy.copy(self.__dict__)
        del pickled_dict['_hmac']
        del pickled_dict['_hmac_256']
        return pickled_dict

    def __setstate__(self, dct):
        self.__dict__ = dct
        self.update_provider(self._provider)