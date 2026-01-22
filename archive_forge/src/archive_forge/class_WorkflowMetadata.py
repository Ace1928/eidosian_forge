from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkflowMetadata(_messages.Message):
    """A Dataproc workflow template resource.

  Enums:
    StateValueValuesEnum: Output only. The workflow state.

  Messages:
    ParametersValue: Map from parameter names to values that were used for
      those parameters.

  Fields:
    clusterName: Output only. The name of the target cluster.
    clusterUuid: Output only. The UUID of target cluster.
    createCluster: Output only. The create cluster operation metadata.
    dagEndTime: Output only. DAG end time, only set for workflows with
      dag_timeout when DAG ends.
    dagStartTime: Output only. DAG start time, only set for workflows with
      dag_timeout when DAG begins.
    dagTimeout: Output only. The timeout duration for the DAG of jobs,
      expressed in seconds (see JSON representation of duration
      (https://developers.google.com/protocol-buffers/docs/proto3#json)).
    deleteCluster: Output only. The delete cluster operation metadata.
    endTime: Output only. Workflow end time.
    graph: Output only. The workflow graph.
    parameters: Map from parameter names to values that were used for those
      parameters.
    startTime: Output only. Workflow start time.
    state: Output only. The workflow state.
    template: Output only. The resource name of the workflow template as
      described in https://cloud.google.com/apis/design/resource_names. For
      projects.regions.workflowTemplates, the resource name of the template
      has the following format:
      projects/{project_id}/regions/{region}/workflowTemplates/{template_id}
      For projects.locations.workflowTemplates, the resource name of the
      template has the following format: projects/{project_id}/locations/{loca
      tion}/workflowTemplates/{template_id}
    version: Output only. The version of template at the time of workflow
      instantiation.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The workflow state.

    Values:
      UNKNOWN: Unused.
      PENDING: The operation has been created.
      RUNNING: The operation is running.
      DONE: The operation is done; either cancelled or completed.
    """
        UNKNOWN = 0
        PENDING = 1
        RUNNING = 2
        DONE = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParametersValue(_messages.Message):
        """Map from parameter names to values that were used for those
    parameters.

    Messages:
      AdditionalProperty: An additional property for a ParametersValue object.

    Fields:
      additionalProperties: Additional properties of type ParametersValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ParametersValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    clusterName = _messages.StringField(1)
    clusterUuid = _messages.StringField(2)
    createCluster = _messages.MessageField('ClusterOperation', 3)
    dagEndTime = _messages.StringField(4)
    dagStartTime = _messages.StringField(5)
    dagTimeout = _messages.StringField(6)
    deleteCluster = _messages.MessageField('ClusterOperation', 7)
    endTime = _messages.StringField(8)
    graph = _messages.MessageField('WorkflowGraph', 9)
    parameters = _messages.MessageField('ParametersValue', 10)
    startTime = _messages.StringField(11)
    state = _messages.EnumField('StateValueValuesEnum', 12)
    template = _messages.StringField(13)
    version = _messages.IntegerField(14, variant=_messages.Variant.INT32)