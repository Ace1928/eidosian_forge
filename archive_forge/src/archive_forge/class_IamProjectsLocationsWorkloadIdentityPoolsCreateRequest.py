from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsWorkloadIdentityPoolsCreateRequest(_messages.Message):
    """A IamProjectsLocationsWorkloadIdentityPoolsCreateRequest object.

  Fields:
    googleIamV1betaWorkloadIdentityPool: A GoogleIamV1betaWorkloadIdentityPool
      resource to be passed as the request body.
    parent: Required. The parent resource to create the pool in. The only
      supported location is `global`.
    workloadIdentityPoolId: Required. The ID to use for the pool, which
      becomes the final component of the resource name. This value should be
      4-32 characters, and may contain the characters [a-z0-9-]. The prefix
      `gcp-` is reserved for use by Google, and may not be specified.
  """
    googleIamV1betaWorkloadIdentityPool = _messages.MessageField('GoogleIamV1betaWorkloadIdentityPool', 1)
    parent = _messages.StringField(2, required=True)
    workloadIdentityPoolId = _messages.StringField(3)