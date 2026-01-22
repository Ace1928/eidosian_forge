from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Conversion(_messages.Message):
    """Conversion defines how to transform an incoming message payload from one
  format to another.

  Fields:
    outputDataFormat: Required. The output data format of the conversion.
  """
    outputDataFormat = _messages.MessageField('DataFormat', 1)