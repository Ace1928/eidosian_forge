from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportInfo(_messages.Message):
    """Message describing import metadata.

  Fields:
    importTime: Output only. [Output only] Import time stamp.
    source: Optional. Resource name of imported source.
    writer: Optional. Writer of imported information.
  """
    importTime = _messages.StringField(1)
    source = _messages.StringField(2)
    writer = _messages.StringField(3)