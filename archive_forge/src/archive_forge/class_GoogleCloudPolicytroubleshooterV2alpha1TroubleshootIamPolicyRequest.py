from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterV2alpha1TroubleshootIamPolicyRequest(_messages.Message):
    """Request for TroubleshootIamPolicy.

  Fields:
    accessTuple: The information to use for checking whether a member has a
      permission for a resource.
  """
    accessTuple = _messages.MessageField('GoogleCloudPolicytroubleshooterV2alpha1AccessTuple', 1)