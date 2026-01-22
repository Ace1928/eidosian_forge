from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ParameterMetadataEnumOption(_messages.Message):
    """ParameterMetadataEnumOption specifies the option shown in the enum form.

  Fields:
    description: Optional. The description to display for the enum option.
    label: Optional. The label to display for the enum option.
    value: Required. The value of the enum option.
  """
    description = _messages.StringField(1)
    label = _messages.StringField(2)
    value = _messages.StringField(3)