from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetadataInfo(_messages.Message):
    """A MetadataInfo object.

  Fields:
    owner: Output only. The owner of the metadata, if it's updated by the
      system.
    updateTime: Output only. Time at which this field was updated.
  """
    owner = _messages.StringField(1)
    updateTime = _messages.StringField(2)