from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Mediation(_messages.Message):
    """Mediation defines different ways to modify the stream.

  Fields:
    bindAttributesAsRawHeaders: Optional. If bind_attributes_as_raw_headers
      set true, we will bind the attributes of an incoming cloud event as raw
      HTTP headers.
    conversion: Optional. Conversion defines the way to convert an incoming
      message payload from one format to another.
    transformation: Optional. Transformation defines the way to transform an
      incoming message.
  """
    bindAttributesAsRawHeaders = _messages.BooleanField(1)
    conversion = _messages.MessageField('Conversion', 2)
    transformation = _messages.MessageField('Transformation', 3)