from __future__ import print_function
import argparse
import contextlib
import datetime
import json
import os
import threading
import warnings
import httplib2
import oauth2client
import oauth2client.client
from oauth2client import service_account
from oauth2client import tools  # for gflags declarations
import six
from six.moves import http_client
from six.moves import urllib
from apitools.base.py import exceptions
from apitools.base.py import util
class GaeAssertionCredentials(oauth2client.client.AssertionCredentials):
    """Assertion credentials for Google App Engine apps."""

    def __init__(self, scopes, **kwds):
        if not util.DetectGae():
            raise exceptions.ResourceUnavailableError('GCE credentials requested outside a GCE instance')
        self._scopes = list(util.NormalizeScopes(scopes))
        super(GaeAssertionCredentials, self).__init__(None, **kwds)

    @classmethod
    def Get(cls, *args, **kwds):
        try:
            return cls(*args, **kwds)
        except exceptions.Error:
            return None

    @classmethod
    def from_json(cls, json_data):
        data = json.loads(json_data)
        return GaeAssertionCredentials(data['_scopes'])

    def _refresh(self, _):
        """Refresh self.access_token.

        Args:
          _: (ignored) A function matching httplib2.Http.request's signature.
        """
        from google.appengine.api import app_identity
        try:
            token, _ = app_identity.get_access_token(self._scopes)
        except app_identity.Error as e:
            raise exceptions.CredentialsError(str(e))
        self.access_token = token

    def sign_blob(self, blob):
        """Cryptographically sign a blob (of bytes).

        This method is provided to support a common interface, but
        the actual key used for a Google Compute Engine service account
        is not available, so it can't be used to sign content.

        Args:
            blob: bytes, Message to be signed.

        Raises:
            NotImplementedError, always.
        """
        raise NotImplementedError('Compute Engine service accounts cannot sign blobs')