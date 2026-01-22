from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamPolicyAnalysisResult(_messages.Message):
    """IAM Policy analysis result, consisting of one IAM policy binding and
  derived access control lists.

  Fields:
    accessControlLists: The access control lists derived from the iam_binding
      that match or potentially match resource and access selectors specified
      in the request.
    attachedResourceFullName: The [full resource
      name](https://cloud.google.com/asset-inventory/docs/resource-name-
      format) of the resource to which the iam_binding policy attaches.
    fullyExplored: Represents whether all analyses on the iam_binding have
      successfully finished.
    iamBinding: The IAM policy binding under analysis.
    identityList: The identity list derived from members of the iam_binding
      that match or potentially match identity selector specified in the
      request.
  """
    accessControlLists = _messages.MessageField('GoogleCloudAssetV1AccessControlList', 1, repeated=True)
    attachedResourceFullName = _messages.StringField(2)
    fullyExplored = _messages.BooleanField(3)
    iamBinding = _messages.MessageField('Binding', 4)
    identityList = _messages.MessageField('GoogleCloudAssetV1IdentityList', 5)