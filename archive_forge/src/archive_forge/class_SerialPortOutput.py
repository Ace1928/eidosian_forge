from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SerialPortOutput(_messages.Message):
    """An instance serial console output.

  Fields:
    contents: [Output Only] The contents of the console output.
    kind: [Output Only] Type of the resource. Always compute#serialPortOutput
      for serial port output.
    next: [Output Only] The position of the next byte of content, regardless
      of whether the content exists, following the output returned in the
      `contents` property. Use this value in the next request as the start
      parameter.
    selfLink: [Output Only] Server-defined URL for this resource.
    start: The starting byte position of the output that was returned. This
      should match the start parameter sent with the request. If the serial
      console output exceeds the size of the buffer (1 MB), older output is
      overwritten by newer content. The output start value will indicate the
      byte position of the output that was returned, which might be different
      than the `start` value that was specified in the request.
  """
    contents = _messages.StringField(1)
    kind = _messages.StringField(2, default='compute#serialPortOutput')
    next = _messages.IntegerField(3)
    selfLink = _messages.StringField(4)
    start = _messages.IntegerField(5)