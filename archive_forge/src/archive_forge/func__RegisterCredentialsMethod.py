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
def _RegisterCredentialsMethod(method, position=None):
    """Register a new method for fetching credentials.

    This new method should be a function with signature:
      client_info, **kwds -> Credentials or None
    This method can be used as a decorator, unless position needs to
    be supplied.

    Note that method must *always* accept arbitrary keyword arguments.

    Args:
      method: New credential-fetching method.
      position: (default: None) Where in the list of methods to
        add this; if None, we append. In all but rare cases,
        this should be either 0 or None.
    Returns:
      method, for use as a decorator.

    """
    if position is None:
        position = len(_CREDENTIALS_METHODS)
    else:
        position = min(position, len(_CREDENTIALS_METHODS))
    _CREDENTIALS_METHODS.insert(position, method)
    return method