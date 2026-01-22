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
class SignedSerializer(object):
    """
    A helper to cryptographically sign arbitrary content using HMAC.

    The serializer accepts arbitrary functions for performing the actual
    serialization and deserialization.

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

    ``serializer``
      An object with two methods: `loads`` and ``dumps``.  The ``loads`` method
      should accept bytes and return a Python object.  The ``dumps`` method
      should accept a Python object and return bytes.  A ``ValueError`` should
      be raised for malformed inputs.  Default: ``None`, which will use a
      derivation of :func:`json.dumps` and ``json.loads``.

    """

    def __init__(self, secret, salt, hashalg='sha512', serializer=None):
        self.salt = salt
        self.secret = secret
        self.hashalg = hashalg
        try:
            self.salted_secret = bytes_(salt or '') + bytes_(secret)
        except UnicodeEncodeError:
            self.salted_secret = bytes_(salt or '', 'utf-8') + bytes_(secret, 'utf-8')
        self.digestmod = lambda string=b'': hashlib.new(self.hashalg, string)
        self.digest_size = self.digestmod().digest_size
        if serializer is None:
            serializer = JSONSerializer()
        self.serializer = serializer

    def dumps(self, appstruct):
        """
        Given an ``appstruct``, serialize and sign the data.

        Returns a bytestring.
        """
        cstruct = self.serializer.dumps(appstruct)
        sig = hmac.new(self.salted_secret, cstruct, self.digestmod).digest()
        return base64.urlsafe_b64encode(sig + cstruct).rstrip(b'=')

    def loads(self, bstruct):
        """
        Given a ``bstruct`` (a bytestring), verify the signature and then
        deserialize and return the deserialized value.

        A ``ValueError`` will be raised if the signature fails to validate.
        """
        try:
            b64padding = b'=' * (-len(bstruct) % 4)
            fstruct = base64.urlsafe_b64decode(bytes_(bstruct) + b64padding)
        except (binascii.Error, TypeError) as e:
            raise ValueError('Badly formed base64 data: %s' % e)
        cstruct = fstruct[self.digest_size:]
        expected_sig = fstruct[:self.digest_size]
        sig = hmac.new(self.salted_secret, bytes_(cstruct), self.digestmod).digest()
        if strings_differ(sig, expected_sig):
            raise ValueError('Invalid signature')
        return self.serializer.loads(cstruct)