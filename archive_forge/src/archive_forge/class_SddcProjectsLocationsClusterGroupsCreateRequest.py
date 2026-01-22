from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SddcProjectsLocationsClusterGroupsCreateRequest(_messages.Message):
    """A SddcProjectsLocationsClusterGroupsCreateRequest object.

  Fields:
    clusterGroup: A ClusterGroup resource to be passed as the request body.
    clusterGroupId: Required. The user-provided ID of the `ClusterGroup` to
      create. This ID must be unique among `ClusterGroup` objects within the
      parent and becomes the final token in the name URI.
    parent: Required. The location (region) and project where the new
      `ClusterGroup` is created. For example, projects/PROJECT-
      NUMBER/locations/us-central1
  """
    clusterGroup = _messages.MessageField('ClusterGroup', 1)
    clusterGroupId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)