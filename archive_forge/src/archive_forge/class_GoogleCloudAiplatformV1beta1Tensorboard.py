from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1Tensorboard(_messages.Message):
    """Tensorboard is a physical database that stores users' training metrics.
  A default Tensorboard is provided in each region of a Google Cloud project.
  If needed users can also create extra Tensorboards in their projects.

  Messages:
    LabelsValue: The labels with user-defined metadata to organize your
      Tensorboards. Label keys and values can be no longer than 64 characters
      (Unicode codepoints), can only contain lowercase letters, numeric
      characters, underscores and dashes. International characters are
      allowed. No more than 64 user labels can be associated with one
      Tensorboard (System labels are excluded). See https://goo.gl/xmQnxf for
      more information and examples of labels. System reserved label keys are
      prefixed with "aiplatform.googleapis.com/" and are immutable.

  Fields:
    blobStoragePathPrefix: Output only. Consumer project Cloud Storage path
      prefix used to store blob data, which can either be a bucket or
      directory. Does not end with a '/'.
    createTime: Output only. Timestamp when this Tensorboard was created.
    description: Description of this Tensorboard.
    displayName: Required. User provided name of this Tensorboard.
    encryptionSpec: Customer-managed encryption key spec for a Tensorboard. If
      set, this Tensorboard and all sub-resources of this Tensorboard will be
      secured by this key.
    etag: Used to perform a consistent read-modify-write updates. If not set,
      a blind "overwrite" update happens.
    isDefault: Used to indicate if the TensorBoard instance is the default
      one. Each project & region can have at most one default TensorBoard
      instance. Creation of a default TensorBoard instance and updating an
      existing TensorBoard instance to be default will mark all other
      TensorBoard instances (if any) as non default.
    labels: The labels with user-defined metadata to organize your
      Tensorboards. Label keys and values can be no longer than 64 characters
      (Unicode codepoints), can only contain lowercase letters, numeric
      characters, underscores and dashes. International characters are
      allowed. No more than 64 user labels can be associated with one
      Tensorboard (System labels are excluded). See https://goo.gl/xmQnxf for
      more information and examples of labels. System reserved label keys are
      prefixed with "aiplatform.googleapis.com/" and are immutable.
    name: Output only. Name of the Tensorboard. Format:
      `projects/{project}/locations/{location}/tensorboards/{tensorboard}`
    runCount: Output only. The number of Runs stored in this Tensorboard.
    updateTime: Output only. Timestamp when this Tensorboard was last updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels with user-defined metadata to organize your Tensorboards.
    Label keys and values can be no longer than 64 characters (Unicode
    codepoints), can only contain lowercase letters, numeric characters,
    underscores and dashes. International characters are allowed. No more than
    64 user labels can be associated with one Tensorboard (System labels are
    excluded). See https://goo.gl/xmQnxf for more information and examples of
    labels. System reserved label keys are prefixed with
    "aiplatform.googleapis.com/" and are immutable.

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
    blobStoragePathPrefix = _messages.StringField(1)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    displayName = _messages.StringField(4)
    encryptionSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1EncryptionSpec', 5)
    etag = _messages.StringField(6)
    isDefault = _messages.BooleanField(7)
    labels = _messages.MessageField('LabelsValue', 8)
    name = _messages.StringField(9)
    runCount = _messages.IntegerField(10, variant=_messages.Variant.INT32)
    updateTime = _messages.StringField(11)