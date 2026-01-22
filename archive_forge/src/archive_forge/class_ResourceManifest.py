from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceManifest(_messages.Message):
    """ResourceManifest represents a single Kubernetes resource to be applied
  to the cluster.

  Fields:
    clusterScoped: Whether the resource provided in the manifest is
      `cluster_scoped`. If unset, the manifest is assumed to be namespace
      scoped. This field is used for REST mapping when applying the resource
      in a cluster.
    manifest: YAML manifest of the resource.
  """
    clusterScoped = _messages.BooleanField(1)
    manifest = _messages.StringField(2)