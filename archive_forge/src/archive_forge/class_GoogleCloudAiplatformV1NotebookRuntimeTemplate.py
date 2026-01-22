from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1NotebookRuntimeTemplate(_messages.Message):
    """A template that specifies runtime configurations such as machine type,
  runtime version, network configurations, etc. Multiple runtimes can be
  created from a runtime template.

  Enums:
    NotebookRuntimeTypeValueValuesEnum: Optional. Immutable. The type of the
      notebook runtime template.

  Messages:
    LabelsValue: The labels with user-defined metadata to organize the
      NotebookRuntimeTemplates. Label keys and values can be no longer than 64
      characters (Unicode codepoints), can only contain lowercase letters,
      numeric characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information and examples of
      labels.

  Fields:
    createTime: Output only. Timestamp when this NotebookRuntimeTemplate was
      created.
    dataPersistentDiskSpec: Optional. The specification of persistent disk
      attached to the runtime as data disk storage.
    description: The description of the NotebookRuntimeTemplate.
    displayName: Required. The display name of the NotebookRuntimeTemplate.
      The name can be up to 128 characters long and can consist of any UTF-8
      characters.
    etag: Used to perform consistent read-modify-write updates. If not set, a
      blind "overwrite" update happens.
    eucConfig: EUC configuration of the NotebookRuntimeTemplate.
    idleShutdownConfig: The idle shutdown configuration of
      NotebookRuntimeTemplate. This config will only be set when idle shutdown
      is enabled.
    isDefault: Output only. The default template to use if not specified.
    labels: The labels with user-defined metadata to organize the
      NotebookRuntimeTemplates. Label keys and values can be no longer than 64
      characters (Unicode codepoints), can only contain lowercase letters,
      numeric characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information and examples of
      labels.
    machineSpec: Optional. Immutable. The specification of a single machine
      for the template.
    name: Output only. The resource name of the NotebookRuntimeTemplate.
    networkSpec: Optional. Network spec.
    notebookRuntimeType: Optional. Immutable. The type of the notebook runtime
      template.
    serviceAccount: The service account that the runtime workload runs as. You
      can use any service account within the same project, but you must have
      the service account user permission to use the instance. If not
      specified, the [Compute Engine default service
      account](https://cloud.google.com/compute/docs/access/service-
      accounts#default_service_account) is used.
    updateTime: Output only. Timestamp when this NotebookRuntimeTemplate was
      most recently updated.
  """

    class NotebookRuntimeTypeValueValuesEnum(_messages.Enum):
        """Optional. Immutable. The type of the notebook runtime template.

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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels with user-defined metadata to organize the
    NotebookRuntimeTemplates. Label keys and values can be no longer than 64
    characters (Unicode codepoints), can only contain lowercase letters,
    numeric characters, underscores and dashes. International characters are
    allowed. See https://goo.gl/xmQnxf for more information and examples of
    labels.

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
    dataPersistentDiskSpec = _messages.MessageField('GoogleCloudAiplatformV1PersistentDiskSpec', 2)
    description = _messages.StringField(3)
    displayName = _messages.StringField(4)
    etag = _messages.StringField(5)
    eucConfig = _messages.MessageField('GoogleCloudAiplatformV1NotebookEucConfig', 6)
    idleShutdownConfig = _messages.MessageField('GoogleCloudAiplatformV1NotebookIdleShutdownConfig', 7)
    isDefault = _messages.BooleanField(8)
    labels = _messages.MessageField('LabelsValue', 9)
    machineSpec = _messages.MessageField('GoogleCloudAiplatformV1MachineSpec', 10)
    name = _messages.StringField(11)
    networkSpec = _messages.MessageField('GoogleCloudAiplatformV1NetworkSpec', 12)
    notebookRuntimeType = _messages.EnumField('NotebookRuntimeTypeValueValuesEnum', 13)
    serviceAccount = _messages.StringField(14)
    updateTime = _messages.StringField(15)