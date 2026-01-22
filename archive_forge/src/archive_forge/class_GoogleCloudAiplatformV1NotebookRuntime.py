from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1NotebookRuntime(_messages.Message):
    """A runtime is a virtual machine allocated to a particular user for a
  particular Notebook file on temporary basis with lifetime limited to 24
  hours.

  Enums:
    HealthStateValueValuesEnum: Output only. The health state of the
      NotebookRuntime.
    NotebookRuntimeTypeValueValuesEnum: Output only. The type of the notebook
      runtime.
    RuntimeStateValueValuesEnum: Output only. The runtime (instance) state of
      the NotebookRuntime.

  Messages:
    LabelsValue: The labels with user-defined metadata to organize your
      NotebookRuntime. Label keys and values can be no longer than 64
      characters (Unicode codepoints), can only contain lowercase letters,
      numeric characters, underscores and dashes. International characters are
      allowed. No more than 64 user labels can be associated with one
      NotebookRuntime (System labels are excluded). See https://goo.gl/xmQnxf
      for more information and examples of labels. System reserved label keys
      are prefixed with "aiplatform.googleapis.com/" and are immutable.
      Following system labels exist for NotebookRuntime: *
      "aiplatform.googleapis.com/notebook_runtime_gce_instance_id": output
      only, its value is the Compute Engine instance id. *
      "aiplatform.googleapis.com/colab_enterprise_entry_service": its value is
      either "bigquery" or "vertex"; if absent, it should be "vertex". This is
      to describe the entry service, either BigQuery or Vertex.

  Fields:
    createTime: Output only. Timestamp when this NotebookRuntime was created.
    description: The description of the NotebookRuntime.
    displayName: Required. The display name of the NotebookRuntime. The name
      can be up to 128 characters long and can consist of any UTF-8
      characters.
    expirationTime: Output only. Timestamp when this NotebookRuntime will be
      expired: 1. System Predefined NotebookRuntime: 24 hours after creation.
      After expiration, system predifined runtime will be deleted. 2. User
      created NotebookRuntime: 6 months after last upgrade. After expiration,
      user created runtime will be stopped and allowed for upgrade.
    healthState: Output only. The health state of the NotebookRuntime.
    isUpgradable: Output only. Whether NotebookRuntime is upgradable.
    labels: The labels with user-defined metadata to organize your
      NotebookRuntime. Label keys and values can be no longer than 64
      characters (Unicode codepoints), can only contain lowercase letters,
      numeric characters, underscores and dashes. International characters are
      allowed. No more than 64 user labels can be associated with one
      NotebookRuntime (System labels are excluded). See https://goo.gl/xmQnxf
      for more information and examples of labels. System reserved label keys
      are prefixed with "aiplatform.googleapis.com/" and are immutable.
      Following system labels exist for NotebookRuntime: *
      "aiplatform.googleapis.com/notebook_runtime_gce_instance_id": output
      only, its value is the Compute Engine instance id. *
      "aiplatform.googleapis.com/colab_enterprise_entry_service": its value is
      either "bigquery" or "vertex"; if absent, it should be "vertex". This is
      to describe the entry service, either BigQuery or Vertex.
    name: Output only. The resource name of the NotebookRuntime.
    networkTags: Optional. The Compute Engine tags to add to runtime (see
      [Tagging instances](https://cloud.google.com/vpc/docs/add-remove-
      network-tags)).
    notebookRuntimeTemplateRef: Output only. The pointer to
      NotebookRuntimeTemplate this NotebookRuntime is created from.
    notebookRuntimeType: Output only. The type of the notebook runtime.
    proxyUri: Output only. The proxy endpoint used to access the
      NotebookRuntime.
    runtimeState: Output only. The runtime (instance) state of the
      NotebookRuntime.
    runtimeUser: Required. The user email of the NotebookRuntime.
    serviceAccount: Output only. The service account that the NotebookRuntime
      workload runs as.
    updateTime: Output only. Timestamp when this NotebookRuntime was most
      recently updated.
    version: Output only. The VM os image version of NotebookRuntime.
  """

    class HealthStateValueValuesEnum(_messages.Enum):
        """Output only. The health state of the NotebookRuntime.

    Values:
      HEALTH_STATE_UNSPECIFIED: Unspecified health state.
      HEALTHY: NotebookRuntime is in healthy state. Applies to ACTIVE state.
      UNHEALTHY: NotebookRuntime is in unhealthy state. Applies to ACTIVE
        state.
    """
        HEALTH_STATE_UNSPECIFIED = 0
        HEALTHY = 1
        UNHEALTHY = 2

    class NotebookRuntimeTypeValueValuesEnum(_messages.Enum):
        """Output only. The type of the notebook runtime.

    Values:
      NOTEBOOK_RUNTIME_TYPE_UNSPECIFIED: Unspecified notebook runtime type,
        NotebookRuntimeType will default to USER_DEFINED.
      USER_DEFINED: runtime or template with coustomized configurations from
        user.
      ONE_CLICK: runtime or template with system defined configurations.
    """
        NOTEBOOK_RUNTIME_TYPE_UNSPECIFIED = 0
        USER_DEFINED = 1
        ONE_CLICK = 2

    class RuntimeStateValueValuesEnum(_messages.Enum):
        """Output only. The runtime (instance) state of the NotebookRuntime.

    Values:
      RUNTIME_STATE_UNSPECIFIED: Unspecified runtime state.
      RUNNING: NotebookRuntime is in running state.
      BEING_STARTED: NotebookRuntime is in starting state.
      BEING_STOPPED: NotebookRuntime is in stopping state.
      STOPPED: NotebookRuntime is in stopped state.
      BEING_UPGRADED: NotebookRuntime is in upgrading state. It is in the
        middle of upgrading process.
      ERROR: NotebookRuntime was unable to start/stop properly.
      INVALID: NotebookRuntime is in invalid state. Cannot be recovered.
    """
        RUNTIME_STATE_UNSPECIFIED = 0
        RUNNING = 1
        BEING_STARTED = 2
        BEING_STOPPED = 3
        STOPPED = 4
        BEING_UPGRADED = 5
        ERROR = 6
        INVALID = 7

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels with user-defined metadata to organize your
    NotebookRuntime. Label keys and values can be no longer than 64 characters
    (Unicode codepoints), can only contain lowercase letters, numeric
    characters, underscores and dashes. International characters are allowed.
    No more than 64 user labels can be associated with one NotebookRuntime
    (System labels are excluded). See https://goo.gl/xmQnxf for more
    information and examples of labels. System reserved label keys are
    prefixed with "aiplatform.googleapis.com/" and are immutable. Following
    system labels exist for NotebookRuntime: *
    "aiplatform.googleapis.com/notebook_runtime_gce_instance_id": output only,
    its value is the Compute Engine instance id. *
    "aiplatform.googleapis.com/colab_enterprise_entry_service": its value is
    either "bigquery" or "vertex"; if absent, it should be "vertex". This is
    to describe the entry service, either BigQuery or Vertex.

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
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    expirationTime = _messages.StringField(4)
    healthState = _messages.EnumField('HealthStateValueValuesEnum', 5)
    isUpgradable = _messages.BooleanField(6)
    labels = _messages.MessageField('LabelsValue', 7)
    name = _messages.StringField(8)
    networkTags = _messages.StringField(9, repeated=True)
    notebookRuntimeTemplateRef = _messages.MessageField('GoogleCloudAiplatformV1NotebookRuntimeTemplateRef', 10)
    notebookRuntimeType = _messages.EnumField('NotebookRuntimeTypeValueValuesEnum', 11)
    proxyUri = _messages.StringField(12)
    runtimeState = _messages.EnumField('RuntimeStateValueValuesEnum', 13)
    runtimeUser = _messages.StringField(14)
    serviceAccount = _messages.StringField(15)
    updateTime = _messages.StringField(16)
    version = _messages.StringField(17)