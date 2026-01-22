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
def _parse_cache_control(headers):
    retval = {}
    if 'cache-control' in headers:
        parts = headers['cache-control'].split(',')
        parts_with_args = [tuple([x.strip().lower() for x in part.split('=', 1)]) for part in parts if -1 != part.find('=')]
        parts_wo_args = [(name.strip().lower(), 1) for name in parts if -1 == name.find('=')]
        retval = dict(parts_with_args + parts_wo_args)
    return retval