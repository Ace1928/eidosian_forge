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
def _bind_write_headers(msg):

    def _write_headers(self):
        for h, v in msg.items():
            print('%s:' % h, end=' ', file=self._fp)
            if isinstance(v, header.Header):
                print(v.encode(maxlinelen=self._maxheaderlen), file=self._fp)
            else:
                headers = header.Header(v, maxlinelen=self._maxheaderlen, charset='utf-8', header_name=h)
                print(headers.encode(), file=self._fp)
        print(file=self._fp)
    return _write_headers