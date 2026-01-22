from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudRunMetadata(_messages.Message):
    """CloudRunMetadata contains information from a Cloud Run deployment.

  Fields:
    job: Output only. The name of the Cloud Run job that is associated with a
      `Rollout`. Format is
      `projects/{project}/locations/{location}/jobs/{job_name}`.
    revision: Output only. The Cloud Run Revision id associated with a
      `Rollout`.
    service: Output only. The name of the Cloud Run Service that is associated
      with a `Rollout`. Format is
      `projects/{project}/locations/{location}/services/{service}`.
    serviceUrls: Output only. The Cloud Run Service urls that are associated
      with a `Rollout`.
  """
    job = _messages.StringField(1)
    revision = _messages.StringField(2)
    service = _messages.StringField(3)
    serviceUrls = _messages.StringField(4, repeated=True)