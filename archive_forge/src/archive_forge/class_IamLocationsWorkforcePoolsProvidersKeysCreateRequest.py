from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamLocationsWorkforcePoolsProvidersKeysCreateRequest(_messages.Message):
    """A IamLocationsWorkforcePoolsProvidersKeysCreateRequest object.

  Fields:
    parent: Required. The provider to create this key in.
    workforcePoolProviderKey: A WorkforcePoolProviderKey resource to be passed
      as the request body.
    workforcePoolProviderKeyId: Required. The ID to use for the key, which
      becomes the final component of the resource name. This value must be
      4-32 characters, and may contain the characters [a-z0-9-].
  """
    parent = _messages.StringField(1, required=True)
    workforcePoolProviderKey = _messages.MessageField('WorkforcePoolProviderKey', 2)
    workforcePoolProviderKeyId = _messages.StringField(3)