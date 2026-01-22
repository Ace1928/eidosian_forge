from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import re
import string
import sys
import wsgiref.util
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import backendinfo
from googlecloudsdk.third_party.appengine._internal import six_subset
@staticmethod
def AssertHeaderNotTooLong(name, value):
    header_length = len(('%s: %s\r\n' % (name, value)).encode('ascii'))
    if header_length >= HttpHeadersDict.MAX_HEADER_LENGTH:
        try:
            max_len = HttpHeadersDict.MAX_HEADER_VALUE_LENGTHS[name]
        except KeyError:
            raise appinfo_errors.InvalidHttpHeaderValue('HTTP header (name + value) is too long.')
        if len(value) > max_len:
            insert = (name, len(value), max_len)
            raise appinfo_errors.InvalidHttpHeaderValue('%r header value has length %d, which exceed the maximum allowed, %d.' % insert)