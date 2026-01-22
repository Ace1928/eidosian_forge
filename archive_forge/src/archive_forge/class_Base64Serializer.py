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
class Base64Serializer(object):
    """ A serializer which uses base64 to encode/decode data"""

    def __init__(self, serializer=None):
        if serializer is None:
            serializer = JSONSerializer()
        self.serializer = serializer

    def dumps(self, appstruct):
        """
        Given an ``appstruct``, serialize and sign the data.

        Returns a bytestring.
        """
        cstruct = self.serializer.dumps(appstruct)
        return base64.urlsafe_b64encode(cstruct)

    def loads(self, bstruct):
        """
        Given a ``bstruct`` (a bytestring), verify the signature and then
        deserialize and return the deserialized value.

        A ``ValueError`` will be raised if the signature fails to validate.
        """
        try:
            cstruct = base64.urlsafe_b64decode(bytes_(bstruct))
        except (binascii.Error, TypeError) as e:
            raise ValueError('Badly formed base64 data: %s' % e)
        return self.serializer.loads(cstruct)