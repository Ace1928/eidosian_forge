from __future__ import absolute_import, division, print_function
from passlib.utils.compat import PY3
import base64
import calendar
import json
import logging; log = logging.getLogger(__name__)
import math
import struct
import sys
import time as _time
import re
from warnings import warn
from passlib import exc
from passlib.exc import TokenError, MalformedTokenError, InvalidTokenError, UsedTokenError
from passlib.utils import (to_unicode, to_bytes, consteq,
from passlib.utils.binary import BASE64_CHARS, b32encode, b32decode
from passlib.utils.compat import (u, unicode, native_string_types, bascii_to_str, int_types, num_types,
from passlib.utils.decor import hybrid_method, memoized_property
from passlib.crypto.digest import lookup_hash, compile_hmac, pbkdf2_hmac
from passlib.hash import pbkdf2_sha256
class TotpMatch(SequenceMixin):
    """
    Object returned by :meth:`TOTP.match` and :meth:`TOTP.verify` on a successful match.

    It can be treated as a sequence of ``(counter, time)``,
    or accessed via the following attributes:

    .. autoattribute:: counter
        :annotation: = 0

    .. autoattribute:: time
        :annotation: = 0

    .. autoattribute:: expected_counter
        :annotation: = 0

    .. autoattribute:: skipped
        :annotation: = 0

    .. autoattribute:: expire_time
        :annotation: = 0

    .. autoattribute:: cache_seconds
        :annotation: = 60

    .. autoattribute:: cache_time
        :annotation: = 0

    This object will always have a ``True`` boolean value.
    """
    totp = None
    counter = 0
    time = 0
    window = 30

    def __init__(self, totp, counter, time, window=30):
        """
        .. warning::
            the constructor signature is an internal detail, and is subject to change.
        """
        self.totp = totp
        self.counter = counter
        self.time = time
        self.window = window

    @memoized_property
    def expected_counter(self):
        """
        Counter value expected for timestamp.
        """
        return self.totp._time_to_counter(self.time)

    @memoized_property
    def skipped(self):
        """
        How many steps were skipped between expected and actual matched counter
        value (may be positive, zero, or negative).
        """
        return self.counter - self.expected_counter

    @memoized_property
    def expire_time(self):
        """Timestamp marking end of period when token is valid"""
        return self.totp._counter_to_time(self.counter + 1)

    @memoized_property
    def cache_seconds(self):
        """
        Number of seconds counter should be cached
        before it's guaranteed to have passed outside of verification window.
        """
        return self.totp.period + self.window

    @memoized_property
    def cache_time(self):
        """
        Timestamp marking when counter has passed outside of verification window.
        """
        return self.expire_time + self.window

    def _as_tuple(self):
        return (self.counter, self.time)

    def __repr__(self):
        args = (self.counter, self.time, self.cache_seconds)
        return '<TotpMatch counter=%d time=%d cache_seconds=%d>' % args