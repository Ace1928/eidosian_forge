import urllib.request
import base64
import bisect
import email
import hashlib
import http.client
import io
import os
import posixpath
import re
import socket
import string
import sys
import time
import tempfile
import contextlib
import warnings
from urllib.error import URLError, HTTPError, ContentTooShortError
from urllib.parse import (
from urllib.response import addinfourl, addclosehook
def _parse_realm(self, header):
    found_challenge = False
    for mo in AbstractBasicAuthHandler.rx.finditer(header):
        scheme, quote, realm = mo.groups()
        if quote not in ['"', "'"]:
            warnings.warn('Basic Auth Realm was unquoted', UserWarning, 3)
        yield (scheme, realm)
        found_challenge = True
    if not found_challenge:
        if header:
            scheme = header.split()[0]
        else:
            scheme = ''
        yield (scheme, None)