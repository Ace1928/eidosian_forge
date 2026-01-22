import collections
import contextlib
import json
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import messages
from apitools.gen import extended_descriptor
from apitools.gen import util
def FixupMessageFields(self):
    for message_type in self.file_descriptor.message_types:
        self._FixupMessage(message_type)