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
def _expires_in(self):
    """Return the number of seconds until this token expires.

        If token_expiry is in the past, this method will return 0, meaning the
        token has already expired.

        If token_expiry is None, this method will return None. Note that
        returning 0 in such a case would not be fair: the token may still be
        valid; we just don't know anything about it.
        """
    if self.token_expiry:
        now = _UTCNOW()
        if self.token_expiry > now:
            time_delta = self.token_expiry - now
            return time_delta.days * 86400 + time_delta.seconds
        else:
            return 0