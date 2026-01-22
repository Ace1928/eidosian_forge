from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RetryJobRequest(_messages.Message):
    """RetryJobRequest is the request object used by `RetryJob`.

  Fields:
    jobId: Required. The job ID for the Job to retry.
    overrideDeployPolicy: Optional. Deploy policies to override. Format is
      `projects/{project}/locations/{location}/deployPolicies/a-z{0,62}`.
    phaseId: Required. The phase ID the Job to retry belongs to.
  """
    jobId = _messages.StringField(1)
    overrideDeployPolicy = _messages.StringField(2, repeated=True)
    phaseId = _messages.StringField(3)