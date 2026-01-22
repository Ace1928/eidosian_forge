from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p2beta1ProductKeyValue(_messages.Message):
    """A product label represented as a key-value pair.

  Fields:
    key: The key of the label attached to the product. Cannot be empty and
      cannot exceed 128 bytes.
    value: The value of the label attached to the product. Cannot be empty and
      cannot exceed 128 bytes.
  """
    key = _messages.StringField(1)
    value = _messages.StringField(2)