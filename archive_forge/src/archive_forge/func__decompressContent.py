import base64
import calendar
import copy
import email
import email.feedparser
from email import header
import email.message
import email.utils
import errno
from gettext import gettext as _
import gzip
from hashlib import md5 as _md5
from hashlib import sha1 as _sha
import hmac
import http.client
import io
import os
import random
import re
import socket
import ssl
import sys
import time
import urllib.parse
import zlib
from . import auth
from .error import *
from .iri2uri import iri2uri
from httplib2 import certs
def _decompressContent(response, new_content):
    content = new_content
    try:
        encoding = response.get('content-encoding', None)
        if encoding in ['gzip', 'deflate']:
            if encoding == 'gzip':
                content = gzip.GzipFile(fileobj=io.BytesIO(new_content)).read()
            if encoding == 'deflate':
                content = zlib.decompress(content, -zlib.MAX_WBITS)
            response['content-length'] = str(len(content))
            response['-content-encoding'] = response['content-encoding']
            del response['content-encoding']
    except (IOError, zlib.error):
        content = ''
        raise FailedToDecompressContent(_('Content purported to be compressed with %s but failed to decompress.') % response.get('content-encoding'), response, content)
    return content