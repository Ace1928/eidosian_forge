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
from oauth2client import util
def _update_query_params(uri, params):
    """Updates a URI with new query parameters.

    Args:
        uri: string, A valid URI, with potential existing query parameters.
        params: dict, A dictionary of query parameters.

    Returns:
        The same URI but with the new query parameters added.
    """
    parts = urllib.parse.urlparse(uri)
    query_params = dict(urllib.parse.parse_qsl(parts.query))
    query_params.update(params)
    new_parts = parts._replace(query=urllib.parse.urlencode(query_params))
    return urllib.parse.urlunparse(new_parts)