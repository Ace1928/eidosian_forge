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
class HasOptionalNestedMessage(messages.Message):
    nested = messages.MessageField(OptionalMessage, 1)
    repeated_nested = messages.MessageField(OptionalMessage, 2, repeated=True)