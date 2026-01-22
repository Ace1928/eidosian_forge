from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamLocationsWorkforcePoolsProvidersPatchRequest(_messages.Message):
    """A IamLocationsWorkforcePoolsProvidersPatchRequest object.

  Fields:
    name: Output only. The resource name of the provider. Format: `locations/{
      location}/workforcePools/{workforce_pool_id}/providers/{provider_id}`
    updateMask: Required. The list of fields to update.
    workforcePoolProvider: A WorkforcePoolProvider resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    updateMask = _messages.StringField(2)
    workforcePoolProvider = _messages.MessageField('WorkforcePoolProvider', 3)