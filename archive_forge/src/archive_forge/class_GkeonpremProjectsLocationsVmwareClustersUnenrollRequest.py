from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsVmwareClustersUnenrollRequest(_messages.Message):
    """A GkeonpremProjectsLocationsVmwareClustersUnenrollRequest object.

  Fields:
    allowMissing: If set to true, and the VMware cluster is not found, the
      request will succeed but no action will be taken on the server and
      return a completed LRO.
    etag: The current etag of the VMware Cluster. If an etag is provided and
      does not match the current etag of the cluster, deletion will be blocked
      and an ABORTED error will be returned.
    force: This is required if the cluster has any associated node pools. When
      set, any child node pools will also be unenrolled.
    name: Required. Name of the VMware user cluster to be unenrolled. Format:
      "projects/{project}/locations/{location}/vmwareClusters/{vmware_cluster}
      "
    validateOnly: Validate the request without actually doing any updates.
  """
    allowMissing = _messages.BooleanField(1)
    etag = _messages.StringField(2)
    force = _messages.BooleanField(3)
    name = _messages.StringField(4, required=True)
    validateOnly = _messages.BooleanField(5)