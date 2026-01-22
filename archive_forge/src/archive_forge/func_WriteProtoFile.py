import collections
import contextlib
import json
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import messages
from apitools.gen import extended_descriptor
from apitools.gen import util
def WriteProtoFile(self, printer):
    """Write the messages file to out as proto."""
    self.Validate()
    extended_descriptor.WriteMessagesFile(self.__file_descriptor, self.__package, self.__client_info.version, printer)