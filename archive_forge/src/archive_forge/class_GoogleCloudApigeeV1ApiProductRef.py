from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ApiProductRef(_messages.Message):
    """A GoogleCloudApigeeV1ApiProductRef object.

  Fields:
    apiproduct: Name of the API product.
    status: Status of the API product. Valid values are `approved` or
      `revoked`.
  """
    apiproduct = _messages.StringField(1)
    status = _messages.StringField(2)