from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatafusionProjectsLocationsRemoveIamPolicyRequest(_messages.Message):
    """A DatafusionProjectsLocationsRemoveIamPolicyRequest object.

  Fields:
    removeIamPolicyRequest: A RemoveIamPolicyRequest resource to be passed as
      the request body.
    resource: Required. The resource on which IAM policy to be removed is
      attached to.
  """
    removeIamPolicyRequest = _messages.MessageField('RemoveIamPolicyRequest', 1)
    resource = _messages.StringField(2, required=True)