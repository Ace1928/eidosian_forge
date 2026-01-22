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
def _ssl_wrap_socket_unsupported(sock, key_file, cert_file, disable_validation, ca_certs, ssl_version, hostname, key_password):
    if not disable_validation:
        raise CertificateValidationUnsupported('SSL certificate validation is not supported without the ssl module installed. To avoid this error, install the ssl module, or explicity disable validation.')
    if key_password:
        raise NotSupportedOnThisPlatform('Certificate with password is not supported.')
    ssl_sock = socket.ssl(sock, key_file, cert_file)
    return httplib.FakeSocket(sock, ssl_sock)