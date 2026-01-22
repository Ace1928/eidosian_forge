from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApproveRolloutRequest(_messages.Message):
    """The request object used by `ApproveRollout`.

  Fields:
    approved: Required. True = approve; false = reject
    overrideDeployPolicy: Optional. Deploy policies to override. Format is
      `projects/{project}/locations/{location}/deployPolicies/a-z{0,62}`.
  """
    approved = _messages.BooleanField(1)
    overrideDeployPolicy = _messages.StringField(2, repeated=True)