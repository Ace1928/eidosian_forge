from __future__ import print_function
import httplib2
import json
import os
from select import select
import stat
from sys import stdin
import time
import webbrowser
from base64 import (
from six.moves.urllib.parse import parse_qs
from lazr.restfulclient.errors import HTTPError
from lazr.restfulclient.authorize.oauth import (
from launchpadlib import uris
def _ssl_certificate_validation_disabled():
    """Whether the user has disabled SSL certificate connection.

    Some testing servers have broken certificates.  Rather than raising an
    error, we allow an environment variable,
    ``LP_DISABLE_SSL_CERTIFICATE_VALIDATION`` to disable the check.
    """
    return bool(os.environ.get('LP_DISABLE_SSL_CERTIFICATE_VALIDATION', False))