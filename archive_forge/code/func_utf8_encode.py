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
def utf8_encode(value):
    """Encode a basestring to UTF-8.

    If the string is unicode encode it to UTF-8, if the string is
    str then assume it's already encoded. Otherwise raise a TypeError.

    :param value: A basestring
    :returns: UTF-8 encoded version of value
    :raises TypeError: If value is not basestring
    """
    if isinstance(value, str):
        return _utf8_encoder(value)[0]
    elif isinstance(value, bytes):
        return value
    else:
        value_cls_name = reflection.get_class_name(value, fully_qualified=False)
        raise TypeError('value must be basestring, not %s' % value_cls_name)