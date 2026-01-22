from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NamedPort(_messages.Message):
    """The named port. For example: <"http", 80>.

  Fields:
    name: The name for this named port. The name must be 1-63 characters long,
      and comply with RFC1035.
    port: The port number, which can be a value between 1 and 65535.
  """
    name = _messages.StringField(1)
    port = _messages.IntegerField(2, variant=_messages.Variant.INT32)