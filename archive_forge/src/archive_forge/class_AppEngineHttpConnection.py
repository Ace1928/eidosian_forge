from __future__ import print_function
import base64
import calendar
import copy
import email
import email.FeedParser
import email.Message
import email.Utils
import errno
import gzip
import httplib
import os
import random
import re
import StringIO
import sys
import time
import urllib
import urlparse
import zlib
import hmac
from gettext import gettext as _
import socket
from httplib2 import auth
from httplib2.error import *
from httplib2 import certs
class AppEngineHttpConnection(httplib.HTTPConnection):
    """Use httplib on App Engine, but compensate for its weirdness.

    The parameters key_file, cert_file, proxy_info, ca_certs,
    disable_ssl_certificate_validation, and ssl_version are all dropped on
    the ground.
    """

    def __init__(self, host, port=None, key_file=None, cert_file=None, strict=None, timeout=None, proxy_info=None, ca_certs=None, disable_ssl_certificate_validation=False, ssl_version=None):
        httplib.HTTPConnection.__init__(self, host, port=port, strict=strict, timeout=timeout)