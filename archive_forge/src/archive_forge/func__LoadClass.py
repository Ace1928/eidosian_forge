import base64
import contextlib
import datetime
import logging
import pprint
import six
from six.moves import http_client
from six.moves import urllib
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
from apitools.base.py import util
def _LoadClass(name, messages_module):
    if name.startswith('message_types.'):
        _, _, classname = name.partition('.')
        return getattr(message_types, classname)
    elif '.' not in name:
        return getattr(messages_module, name)
    else:
        raise exceptions.GeneratedClientError('Unknown class %s' % name)