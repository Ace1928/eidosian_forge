from __future__ import absolute_import, unicode_literals
import logging
from itertools import chain
from oauthlib.common import add_params_to_uri
from oauthlib.uri_validate import is_absolute_uri
from oauthlib.oauth2.rfc6749 import errors, utils
from ..request_validator import RequestValidator
def _get_default_headers(self):
    """Create default headers for grant responses."""
    return {'Content-Type': 'application/json', 'Cache-Control': 'no-store', 'Pragma': 'no-cache'}