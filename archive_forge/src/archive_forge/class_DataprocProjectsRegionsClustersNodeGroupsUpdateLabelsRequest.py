from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsRegionsClustersNodeGroupsUpdateLabelsRequest(_messages.Message):
    """A DataprocProjectsRegionsClustersNodeGroupsUpdateLabelsRequest object.

  Fields:
    name: Required. The name of the node group for updating the labels.
      Format: projects/{project}/regions/{region}/clusters/{cluster}/nodeGroup
      s/{nodeGroup}
    updateLabelsNodeGroupRequest: A UpdateLabelsNodeGroupRequest resource to
      be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    updateLabelsNodeGroupRequest = _messages.MessageField('UpdateLabelsNodeGroupRequest', 2)