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
Return the length of this response.

        We expose this as an attribute since using len() directly can fail
        for responses larger than sys.maxint.

        Returns:
          Response length (as int or long)
        