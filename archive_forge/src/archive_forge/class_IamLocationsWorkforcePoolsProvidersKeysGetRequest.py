from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamLocationsWorkforcePoolsProvidersKeysGetRequest(_messages.Message):
    """A IamLocationsWorkforcePoolsProvidersKeysGetRequest object.

  Fields:
    name: Required. The name of the key to retrieve.
  """
    name = _messages.StringField(1, required=True)