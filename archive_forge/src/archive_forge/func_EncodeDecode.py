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
def EncodeDecode(self, encoded, expected_message):
    message = self.PROTOLIB.decode_message(type(expected_message), encoded)
    self.assertEquals(expected_message, message)
    self.CompareEncoded(encoded, self.PROTOLIB.encode_message(message))