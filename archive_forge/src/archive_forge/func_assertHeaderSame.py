import cgi
import datetime
import inspect
import os
import re
import socket
import types
import unittest
import six
from six.moves import range  # pylint: disable=redefined-builtin
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
def assertHeaderSame(self, header1, header2):
    """Check that two HTTP headers are the same.

        Args:
          header1: Header value string 1.
          header2: header value string 2.
        """
    value1, params1 = cgi.parse_header(header1)
    value2, params2 = cgi.parse_header(header2)
    self.assertEqual(value1, value2)
    self.assertEqual(params1, params2)