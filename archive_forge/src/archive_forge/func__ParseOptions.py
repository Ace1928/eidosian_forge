import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
def _ParseOptions(message, string):
    """Parses serialized options.

  This helper function is used to parse serialized options in generated
  proto2 files. It must not be used outside proto2.
  """
    message.ParseFromString(string)
    return message