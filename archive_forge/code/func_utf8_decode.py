import abc
import codecs
import os.path
import random
import re
import sys
import uuid
import weakref
import ldap.controls
import ldap.filter
import ldappool
from oslo_log import log
from oslo_utils import reflection
from keystone.common import driver_hints
from keystone import exception
from keystone.i18n import _
def utf8_decode(value):
    """Decode a from UTF-8 into unicode.

    If the value is a binary string assume it's UTF-8 encoded and decode
    it into a unicode string. Otherwise convert the value from its
    type into a unicode string.

    :param value: value to be returned as unicode
    :returns: value as unicode
    :raises UnicodeDecodeError: for invalid UTF-8 encoding
    """
    if isinstance(value, bytes):
        try:
            return _utf8_decoder(value)[0]
        except UnicodeDecodeError:
            uuid_byte_string_length = 16
            if len(value) == uuid_byte_string_length:
                return str(uuid.UUID(bytes_le=value))
            else:
                raise
    return str(value)