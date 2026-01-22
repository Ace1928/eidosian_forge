import json
import os.path
import shutil
import tempfile
import unittest
import mock
import six
from apitools.base.py import credentials_lib
from apitools.base.py import util
class MetadataMock(object):

    def __init__(self, scopes=None, service_account_name=None):
        self._scopes = scopes or ['scope1']
        self._sa = service_account_name or 'default'

    def __call__(self, request_url):
        if request_url.endswith('scopes'):
            return six.StringIO(''.join(self._scopes))
        elif request_url.endswith('service-accounts'):
            return six.StringIO(self._sa)
        elif request_url.endswith('/service-accounts/%s/token' % self._sa):
            return six.StringIO('{"access_token": "token"}')
        self.fail('Unexpected HTTP request to %s' % request_url)