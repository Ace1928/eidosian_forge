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
def _http_post(url, headers, params):
    """POST to ``url`` with ``headers`` and a body of urlencoded ``params``.

    Wraps it up to make sure we avoid the SSL certificate validation if our
    environment tells us to.  Also, raises an error on non-200 statuses.
    """
    cert_disabled = _ssl_certificate_validation_disabled()
    response, content = httplib2.Http(disable_ssl_certificate_validation=cert_disabled).request(url, method='POST', headers=headers, body=urlencode(params))
    if response.status != 200:
        raise HTTPError(response, content)
    return (response, content)