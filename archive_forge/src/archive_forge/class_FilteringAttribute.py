from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FilteringAttribute(_messages.Message):
    """A representation of the FilteringAttribute resource. Filtering
  attributes are per event type.

  Fields:
    attribute: Output only. Attribute used for filtering the event type.
    description: Output only. Description of the purpose of the attribute.
    pathPatternSupported: Output only. If true, the attribute accepts matching
      expressions in the Eventarc PathPattern format.
    required: Output only. If true, the triggers for this provider should
      always specify a filter on these attributes. Trigger creation will fail
      otherwise.
  """
    attribute = _messages.StringField(1)
    description = _messages.StringField(2)
    pathPatternSupported = _messages.BooleanField(3)
    required = _messages.BooleanField(4)