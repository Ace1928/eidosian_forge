import base64
import binascii
import hashlib
import hmac
import json
from datetime import (
import re
import string
import time
import warnings
from webob.compat import (
from webob.util import strings_differ
class SignedCookieProfile(CookieProfile):
    """
    A helper for generating cookies that are signed to prevent tampering.

    By default this will create a single cookie, given a value it will
    serialize it, then use HMAC to cryptographically sign the data. Finally
    the result is base64-encoded for transport. This way a remote user can
    not tamper with the value without uncovering the secret/salt used.

    ``secret``
      A string which is used to sign the cookie. The secret should be at
      least as long as the block size of the selected hash algorithm. For
      ``sha512`` this would mean a 512 bit (64 character) secret.

    ``salt``
      A namespace to avoid collisions between different uses of a shared
      secret.

    ``hashalg``
      The HMAC digest algorithm to use for signing. The algorithm must be
      supported by the :mod:`hashlib` library. Default: ``'sha512'``.

    ``cookie_name``
      The name of the cookie used for sessioning. Default: ``'session'``.

    ``max_age``
      The maximum age of the cookie used for sessioning (in seconds).
      Default: ``None`` (browser scope).

    ``secure``
      The 'secure' flag of the session cookie. Default: ``False``.

    ``httponly``
      Hide the cookie from Javascript by setting the 'HttpOnly' flag of the
      session cookie. Default: ``False``.

    ``samesite``
      The 'SameSite' attribute of the cookie, can be either ``b"strict"``,
      ``b"lax"``, ``b"none"``, or ``None``.

    ``path``
      The path used for the session cookie. Default: ``'/'``.

    ``domains``
      The domain(s) used for the session cookie. Default: ``None`` (no domain).
      Can be passed an iterable containing multiple domains, this will set
      multiple cookies one for each domain.

    ``serializer``
      An object with two methods: `loads`` and ``dumps``.  The ``loads`` method
      should accept bytes and return a Python object.  The ``dumps`` method
      should accept a Python object and return bytes.  A ``ValueError`` should
      be raised for malformed inputs.  Default: ``None`, which will use a
      derivation of :func:`json.dumps` and ``json.loads``.
    """

    def __init__(self, secret, salt, cookie_name, secure=False, max_age=None, httponly=False, samesite=None, path='/', domains=None, hashalg='sha512', serializer=None):
        self.secret = secret
        self.salt = salt
        self.hashalg = hashalg
        self.original_serializer = serializer
        signed_serializer = SignedSerializer(secret, salt, hashalg, serializer=self.original_serializer)
        CookieProfile.__init__(self, cookie_name, secure=secure, max_age=max_age, httponly=httponly, samesite=samesite, path=path, domains=domains, serializer=signed_serializer)

    def bind(self, request):
        """ Bind a request to a copy of this instance and return it"""
        selfish = SignedCookieProfile(self.secret, self.salt, self.cookie_name, self.secure, self.max_age, self.httponly, self.samesite, self.path, self.domains, self.hashalg, self.original_serializer)
        selfish.request = request
        return selfish