import base64
import json
import linecache
import logging
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections import namedtuple
from copy import copy
from decimal import Decimal
from numbers import Real
from datetime import datetime
from functools import partial
import sentry_sdk
from sentry_sdk._compat import PY2, PY33, PY37, implements_str, text_type, urlparse
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import DEFAULT_MAX_VALUE_LENGTH
@implements_str
class Dsn(object):
    """Represents a DSN."""

    def __init__(self, value):
        if isinstance(value, Dsn):
            self.__dict__ = dict(value.__dict__)
            return
        parts = urlparse.urlsplit(text_type(value))
        if parts.scheme not in ('http', 'https'):
            raise BadDsn('Unsupported scheme %r' % parts.scheme)
        self.scheme = parts.scheme
        if parts.hostname is None:
            raise BadDsn('Missing hostname')
        self.host = parts.hostname
        if parts.port is None:
            self.port = self.scheme == 'https' and 443 or 80
        else:
            self.port = parts.port
        if not parts.username:
            raise BadDsn('Missing public key')
        self.public_key = parts.username
        self.secret_key = parts.password
        path = parts.path.rsplit('/', 1)
        try:
            self.project_id = text_type(int(path.pop()))
        except (ValueError, TypeError):
            raise BadDsn('Invalid project in DSN (%r)' % (parts.path or '')[1:])
        self.path = '/'.join(path) + '/'

    @property
    def netloc(self):
        """The netloc part of a DSN."""
        rv = self.host
        if (self.scheme, self.port) not in (('http', 80), ('https', 443)):
            rv = '%s:%s' % (rv, self.port)
        return rv

    def to_auth(self, client=None):
        """Returns the auth info object for this dsn."""
        return Auth(scheme=self.scheme, host=self.netloc, path=self.path, project_id=self.project_id, public_key=self.public_key, secret_key=self.secret_key, client=client)

    def __str__(self):
        return '%s://%s%s@%s%s%s' % (self.scheme, self.public_key, self.secret_key and '@' + self.secret_key or '', self.netloc, self.path, self.project_id)