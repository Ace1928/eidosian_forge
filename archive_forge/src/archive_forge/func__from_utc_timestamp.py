import os
import sys
import time
import errno
import base64
import logging
import datetime
import urllib.parse
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from libcloud.utils.py3 import b, httplib, urlparse, urlencode
from libcloud.common.base import BaseDriver, JsonResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, ProviderError
from libcloud.utils.connection import get_response_object
def _from_utc_timestamp(timestamp):
    """
    Return datetime obj where date and time are pulled from timestamp string.
    """
    return datetime.datetime.strptime(timestamp, UTC_TIMESTAMP_FORMAT)