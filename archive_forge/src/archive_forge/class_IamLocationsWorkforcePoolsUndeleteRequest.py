from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamLocationsWorkforcePoolsUndeleteRequest(_messages.Message):
    """A IamLocationsWorkforcePoolsUndeleteRequest object.

  Fields:
    name: Required. The name of the pool to undelete. Format:
      `locations/{location}/workforcePools/{workforce_pool_id}`
    undeleteWorkforcePoolRequest: A UndeleteWorkforcePoolRequest resource to
      be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    undeleteWorkforcePoolRequest = _messages.MessageField('UndeleteWorkforcePoolRequest', 2)