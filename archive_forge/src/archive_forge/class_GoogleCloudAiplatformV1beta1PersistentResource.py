from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PersistentResource(_messages.Message):
    """Represents long-lasting resources that are dedicated to users to runs
  custom workloads. A PersistentResource can have multiple node pools and each
  node pool can have its own machine spec.

  Enums:
    StateValueValuesEnum: Output only. The detailed state of a Study.

  Messages:
    LabelsValue: Optional. The labels with user-defined metadata to organize
      PersistentResource. Label keys and values can be no longer than 64
      characters (Unicode codepoints), can only contain lowercase letters,
      numeric characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information and examples of
      labels.

  Fields:
    createTime: Output only. Time when the PersistentResource was created.
    displayName: Optional. The display name of the PersistentResource. The
      name can be up to 128 characters long and can consist of any UTF-8
      characters.
    encryptionSpec: Optional. Customer-managed encryption key spec for a
      PersistentResource. If set, this PersistentResource and all sub-
      resources of this PersistentResource will be secured by this key.
    error: Output only. Only populated when persistent resource's state is
      `STOPPING` or `ERROR`.
    labels: Optional. The labels with user-defined metadata to organize
      PersistentResource. Label keys and values can be no longer than 64
      characters (Unicode codepoints), can only contain lowercase letters,
      numeric characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information and examples of
      labels.
    name: Immutable. Resource name of a PersistentResource.
    network: Optional. The full name of the Compute Engine
      [network](/compute/docs/networks-and-firewalls#networks) to peered with
      Vertex AI to host the persistent resources. For example,
      `projects/12345/global/networks/myVPC`.
      [Format](/compute/docs/reference/rest/v1/networks/insert) is of the form
      `projects/{project}/global/networks/{network}`. Where {project} is a
      project number, as in `12345`, and {network} is a network name. To
      specify this field, you must have already [configured VPC Network
      Peering for Vertex AI](https://cloud.google.com/vertex-
      ai/docs/general/vpc-peering). If this field is left unspecified, the
      resources aren't peered with any network.
    reservedIpRanges: Optional. A list of names for the reserved IP ranges
      under the VPC network that can be used for this persistent resource. If
      set, we will deploy the persistent resource within the provided IP
      ranges. Otherwise, the persistent resource is deployed to any IP ranges
      under the provided VPC network. Example: ['vertex-ai-ip-range'].
    resourcePools: Required. The spec of the pools of different resources.
    resourceRuntime: Output only. Runtime information of the Persistent
      Resource.
    resourceRuntimeSpec: Optional. Persistent Resource runtime spec. For
      example, used for Ray cluster configuration.
    startTime: Output only. Time when the PersistentResource for the first
      time entered the `RUNNING` state.
    state: Output only. The detailed state of a Study.
    updateTime: Output only. Time when the PersistentResource was most
      recently updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The detailed state of a Study.

    Values:
      STATE_UNSPECIFIED: Not set.
      PROVISIONING: The PROVISIONING state indicates the persistent resources
        is being created.
      RUNNING: The RUNNING state indicates the persistent resource is healthy
        and fully usable.
      STOPPING: The STOPPING state indicates the persistent resource is being
        deleted.
      ERROR: The ERROR state indicates the persistent resource may be
        unusable. Details can be found in the `error` field.
      REBOOTING: The REBOOTING state indicates the persistent resource is
        being rebooted (PR is not available right now but is expected to be
        ready again later).
      UPDATING: The UPDATING state indicates the persistent resource is being
        updated.
    """
        STATE_UNSPECIFIED = 0
        PROVISIONING = 1
        RUNNING = 2
        STOPPING = 3
        ERROR = 4
        REBOOTING = 5
        UPDATING = 6

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. The labels with user-defined metadata to organize
    PersistentResource. Label keys and values can be no longer than 64
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
    displayName = _messages.StringField(2)
    encryptionSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1EncryptionSpec', 3)
    error = _messages.MessageField('GoogleRpcStatus', 4)
    labels = _messages.MessageField('LabelsValue', 5)
    name = _messages.StringField(6)
    network = _messages.StringField(7)
    reservedIpRanges = _messages.StringField(8, repeated=True)
    resourcePools = _messages.MessageField('GoogleCloudAiplatformV1beta1ResourcePool', 9, repeated=True)
    resourceRuntime = _messages.MessageField('GoogleCloudAiplatformV1beta1ResourceRuntime', 10)
    resourceRuntimeSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1ResourceRuntimeSpec', 11)
    startTime = _messages.StringField(12)
    state = _messages.EnumField('StateValueValuesEnum', 13)
    updateTime = _messages.StringField(14)