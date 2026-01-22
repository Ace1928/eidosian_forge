from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1Processor(_messages.Message):
    """The first-class citizen for Document AI. Each processor defines how to
  extract structural information from a document.

  Enums:
    StateValueValuesEnum: Output only. The state of the processor.

  Fields:
    createTime: The time the processor was created.
    defaultProcessorVersion: The default processor version.
    displayName: The display name of the processor.
    kmsKeyName: The [KMS key](https://cloud.google.com/security-key-
      management) used for encryption and decryption in CMEK scenarios.
    name: Output only. Immutable. The resource name of the processor. Format:
      `projects/{project}/locations/{location}/processors/{processor}`
    processEndpoint: Output only. Immutable. The http endpoint that can be
      called to invoke processing.
    processorVersionAliases: Output only. The processor version aliases.
    state: Output only. The state of the processor.
    type: The processor type, such as: `OCR_PROCESSOR`, `INVOICE_PROCESSOR`.
      To get a list of processor types, see FetchProcessorTypes.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the processor.

    Values:
      STATE_UNSPECIFIED: The processor is in an unspecified state.
      ENABLED: The processor is enabled, i.e., has an enabled version which
        can currently serve processing requests and all the feature
        dependencies have been successfully initialized.
      DISABLED: The processor is disabled.
      ENABLING: The processor is being enabled, will become `ENABLED` if
        successful.
      DISABLING: The processor is being disabled, will become `DISABLED` if
        successful.
      CREATING: The processor is being created, will become either `ENABLED`
        (for successful creation) or `FAILED` (for failed ones). Once a
        processor is in this state, it can then be used for document
        processing, but the feature dependencies of the processor might not be
        fully created yet.
      FAILED: The processor failed during creation or initialization of
        feature dependencies. The user should delete the processor and
        recreate one as all the functionalities of the processor are disabled.
      DELETING: The processor is being deleted, will be removed if successful.
    """
        STATE_UNSPECIFIED = 0
        ENABLED = 1
        DISABLED = 2
        ENABLING = 3
        DISABLING = 4
        CREATING = 5
        FAILED = 6
        DELETING = 7
    createTime = _messages.StringField(1)
    defaultProcessorVersion = _messages.StringField(2)
    displayName = _messages.StringField(3)
    kmsKeyName = _messages.StringField(4)
    name = _messages.StringField(5)
    processEndpoint = _messages.StringField(6)
    processorVersionAliases = _messages.MessageField('GoogleCloudDocumentaiV1ProcessorVersionAlias', 7, repeated=True)
    state = _messages.EnumField('StateValueValuesEnum', 8)
    type = _messages.StringField(9)