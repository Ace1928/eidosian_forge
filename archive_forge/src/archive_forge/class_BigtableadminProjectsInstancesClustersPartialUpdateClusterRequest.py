from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesClustersPartialUpdateClusterRequest(_messages.Message):
    """A BigtableadminProjectsInstancesClustersPartialUpdateClusterRequest
  object.

  Fields:
    cluster: A Cluster resource to be passed as the request body.
    name: The unique name of the cluster. Values are of the form
      `projects/{project}/instances/{instance}/clusters/a-z*`.
    updateMask: Required. The subset of Cluster fields which should be
      replaced.
  """
    cluster = _messages.MessageField('Cluster', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)