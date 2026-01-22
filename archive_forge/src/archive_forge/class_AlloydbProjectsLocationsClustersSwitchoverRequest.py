from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AlloydbProjectsLocationsClustersSwitchoverRequest(_messages.Message):
    """A AlloydbProjectsLocationsClustersSwitchoverRequest object.

  Fields:
    name: Required. The name of the resource. For the required format, see the
      comment on the Cluster.name field
    switchoverClusterRequest: A SwitchoverClusterRequest resource to be passed
      as the request body.
  """
    name = _messages.StringField(1, required=True)
    switchoverClusterRequest = _messages.MessageField('SwitchoverClusterRequest', 2)