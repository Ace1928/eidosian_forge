from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesClustersCreateRequest(_messages.Message):
    """A BigtableadminProjectsInstancesClustersCreateRequest object.

  Fields:
    cluster: A Cluster resource to be passed as the request body.
    clusterId: Required. The ID to be used when referring to the new cluster
      within its instance, e.g., just `mycluster` rather than
      `projects/myproject/instances/myinstance/clusters/mycluster`.
    parent: Required. The unique name of the instance in which to create the
      new cluster. Values are of the form
      `projects/{project}/instances/{instance}`.
  """
    cluster = _messages.MessageField('Cluster', 1)
    clusterId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)