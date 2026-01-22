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
def __EncodePrettyPrint(self, query_info):
    if not query_info.pop('prettyPrint', True):
        query_info['prettyPrint'] = 0
    if not query_info.pop('pp', True):
        query_info['pp'] = 0
    return query_info