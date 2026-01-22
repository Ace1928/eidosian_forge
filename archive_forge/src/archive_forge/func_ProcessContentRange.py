import collections
import contextlib
import logging
import socket
import time
import httplib2
import six
from six.moves import http_client
from six.moves.urllib import parse
from apitools.base.py import exceptions
from apitools.base.py import util
def ProcessContentRange(content_range):
    _, _, range_spec = content_range.partition(' ')
    byte_range, _, _ = range_spec.partition('/')
    start, _, end = byte_range.partition('-')
    return int(end) - int(start) + 1