from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlProjectsJobsPatchRequest(_messages.Message):
    """A MlProjectsJobsPatchRequest object.

  Fields:
    googleCloudMlV1Job: A GoogleCloudMlV1Job resource to be passed as the
      request body.
    name: Required. The job name.
    updateMask: Required. Specifies the path, relative to `Job`, of the field
      to update. To adopt etag mechanism, include `etag` field in the mask,
      and include the `etag` value in your job resource. For example, to
      change the labels of a job, the `update_mask` parameter would be
      specified as `labels`, `etag`, and the `PATCH` request body would
      specify the new value, as follows: { "labels": { "owner": "Google",
      "color": "Blue" } "etag": "33a64df551425fcc55e4d42a148795d9f25f89d4" }
      If `etag` matches the one on the server, the labels of the job will be
      replaced with the given ones, and the server end `etag` will be
      recalculated. Currently the only supported update masks are `labels` and
      `etag`.
  """
    googleCloudMlV1Job = _messages.MessageField('GoogleCloudMlV1Job', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)