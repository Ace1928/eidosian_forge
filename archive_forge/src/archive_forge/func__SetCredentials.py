import base64
import contextlib
import datetime
import logging
import pprint
import six
from six.moves import http_client
from six.moves import urllib
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
from apitools.base.py import util
def _SetCredentials(self, **kwds):
    """Fetch credentials, and set them for this client.

        Note that we can't simply return credentials, since creating them
        may involve side-effecting self.

        Args:
          **kwds: Additional keyword arguments are passed on to GetCredentials.

        Returns:
          None. Sets self._credentials.
        """
    args = {'api_key': self._API_KEY, 'client': self, 'client_id': self._CLIENT_ID, 'client_secret': self._CLIENT_SECRET, 'package_name': self._PACKAGE, 'scopes': self._SCOPES, 'user_agent': self._USER_AGENT}
    args.update(kwds)
    from apitools.base.py import credentials_lib
    self._credentials = credentials_lib.GetCredentials(**args)