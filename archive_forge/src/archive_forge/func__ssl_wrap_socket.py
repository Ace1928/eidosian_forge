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
def _ssl_wrap_socket(sock, key_file, cert_file, disable_validation, ca_certs, ssl_version, hostname, key_password):
    if disable_validation:
        cert_reqs = ssl.CERT_NONE
    else:
        cert_reqs = ssl.CERT_REQUIRED
    if ssl_version is None:
        ssl_version = ssl.PROTOCOL_SSLv23
    if hasattr(ssl, 'SSLContext'):
        context = ssl.SSLContext(ssl_version)
        context.verify_mode = cert_reqs
        context.check_hostname = cert_reqs != ssl.CERT_NONE
        if cert_file:
            if key_password:
                context.load_cert_chain(cert_file, key_file, key_password)
            else:
                context.load_cert_chain(cert_file, key_file)
        if ca_certs:
            context.load_verify_locations(ca_certs)
        return context.wrap_socket(sock, server_hostname=hostname)
    else:
        if key_password:
            raise NotSupportedOnThisPlatform('Certificate with password is not supported.')
        return ssl.wrap_socket(sock, keyfile=key_file, certfile=cert_file, cert_reqs=cert_reqs, ca_certs=ca_certs, ssl_version=ssl_version)