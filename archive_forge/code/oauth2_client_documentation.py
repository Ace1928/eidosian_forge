from __future__ import absolute_import
import datetime
import errno
from hashlib import sha1
import json
import logging
import os
import socket
import tempfile
import threading
import boto
import httplib2
import oauth2client.client
import oauth2client.service_account
from google_reauth import reauth_creds
import retry_decorator.retry_decorator
import six
from six import BytesIO
from six.moves import urllib
Whether the access token needs to be refreshed.

    Args:
      time_delta: refresh access token when it expires within time_delta secs.

    Returns:
      True if the token is expired or about to expire, False if the
      token should be expected to work.  Note that the token may still
      be rejected, e.g. if it has been revoked server-side.
    