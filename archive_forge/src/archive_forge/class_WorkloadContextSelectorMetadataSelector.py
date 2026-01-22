from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadContextSelectorMetadataSelector(_messages.Message):
    """This message type exists as opposed to using a map to support additional
  fields in the future such as priority.

  Fields:
    key: Required. The metadata field being selected on
    value: Required. The value for this metadata field to be compared with
  """
    key = _messages.StringField(1)
    value = _messages.StringField(2)