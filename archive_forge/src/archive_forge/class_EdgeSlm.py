from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EdgeSlm(_messages.Message):
    """EdgeSlm represents an SLM instance which manages the lifecycle of edge
  components installed on Workload clusters managed by an Orchestration
  Cluster.

  Enums:
    StateValueValuesEnum: Output only. State of the EdgeSlm resource.
    WorkloadClusterTypeValueValuesEnum: Optional. Type of workload cluster for
      which an EdgeSLM resource is created.

  Messages:
    LabelsValue: Optional. Labels as key value pairs. The key and value should
      contain characters which are UTF-8 compliant and less than 50
      characters.

  Fields:
    createTime: Output only. [Output only] Create time stamp.
    labels: Optional. Labels as key value pairs. The key and value should
      contain characters which are UTF-8 compliant and less than 50
      characters.
    name: Name of the EdgeSlm resource.
    orchestrationCluster: Immutable. Reference to the orchestration cluster on
      which templates for this resources will be applied. This should be of
      format projects/{project}/locations/{location}/orchestrationClusters/{or
      chestration_cluster}.
    state: Output only. State of the EdgeSlm resource.
    tnaVersion: Output only. Provides the active TNA version for this
      resource.
    updateTime: Output only. [Output only] Update time stamp.
    workloadClusterType: Optional. Type of workload cluster for which an
      EdgeSLM resource is created.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the EdgeSlm resource.

    Values:
      STATE_UNSPECIFIED: Unspecified state.
      CREATING: EdgeSlm is being created.
      ACTIVE: EdgeSlm has been created and is ready for use.
      DELETING: EdgeSlm is being deleted.
      FAILED: EdgeSlm encountered an error and is in an indeterministic state.
        User can still initiate a delete operation on this state.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        DELETING = 3
        FAILED = 4

    class WorkloadClusterTypeValueValuesEnum(_messages.Enum):
        """Optional. Type of workload cluster for which an EdgeSLM resource is
    created.

    Values:
      WORKLOAD_CLUSTER_TYPE_UNSPECIFIED: Unspecified workload cluster.
      GDCE: Workload cluster is a GDCE cluster.
      GKE: Workload cluster is a GKE cluster.
    """
        WORKLOAD_CLUSTER_TYPE_UNSPECIFIED = 0
        GDCE = 1
        GKE = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels as key value pairs. The key and value should contain
    characters which are UTF-8 compliant and less than 50 characters.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    labels = _messages.MessageField('LabelsValue', 2)
    name = _messages.StringField(3)
    orchestrationCluster = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)
    tnaVersion = _messages.StringField(6)
    updateTime = _messages.StringField(7)
    workloadClusterType = _messages.EnumField('WorkloadClusterTypeValueValuesEnum', 8)