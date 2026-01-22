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
def _generate_refresh_request_body(self):
    assertion = self._generate_assertion()
    body = urllib.parse.urlencode({'assertion': assertion, 'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer'})
    return body