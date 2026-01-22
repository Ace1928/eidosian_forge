from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RedisProjectsLocationsClustersGetCertificateAuthorityRequest(_messages.Message):
    """A RedisProjectsLocationsClustersGetCertificateAuthorityRequest object.

  Fields:
    name: Required. Redis cluster certificate authority resource name using
      the form: `projects/{project_id}/locations/{location_id}/clusters/{clust
      er_id}/certificateAuthority` where `location_id` refers to a GCP region.
  """
    name = _messages.StringField(1, required=True)