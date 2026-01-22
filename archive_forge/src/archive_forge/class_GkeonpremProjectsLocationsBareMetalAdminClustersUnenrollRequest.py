from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsBareMetalAdminClustersUnenrollRequest(_messages.Message):
    """A GkeonpremProjectsLocationsBareMetalAdminClustersUnenrollRequest
  object.

  Fields:
    allowMissing: If set to true, and the bare metal admin cluster is not
      found, the request will succeed but no action will be taken on the
      server and return a completed LRO.
    etag: The current etag of the bare metal admin cluster. If an etag is
      provided and does not match the current etag of the cluster, deletion
      will be blocked and an ABORTED error will be returned.
    ignoreErrors: If set to true, the unenrollment of a bare metal admin
      cluster resource will succeed even if errors occur during unenrollment.
      This parameter can be used when you want to unenroll admin cluster
      resource and the on-prem admin cluster is disconnected / unreachable.
      WARNING: Using this parameter when your admin cluster still exists may
      result in a deleted GCP admin cluster but existing resourcelink in on-
      prem admin cluster and membership.
    name: Required. Name of the bare metal admin cluster to be unenrolled.
      Format: "projects/{project}/locations/{location}/bareMetalAdminClusters/
      {cluster}"
    validateOnly: Validate the request without actually doing any updates.
  """
    allowMissing = _messages.BooleanField(1)
    etag = _messages.StringField(2)
    ignoreErrors = _messages.BooleanField(3)
    name = _messages.StringField(4, required=True)
    validateOnly = _messages.BooleanField(5)