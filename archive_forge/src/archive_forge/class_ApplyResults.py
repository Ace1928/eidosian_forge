from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApplyResults(_messages.Message):
    """Outputs and artifacts from applying a deployment.

  Messages:
    OutputsValue: Map of output name to output info.

  Fields:
    artifacts: Location of artifacts (e.g. logs) in Google Cloud Storage.
      Format: `gs://{bucket}/{object}`
    content: Location of a blueprint copy and other manifests in Google Cloud
      Storage. Format: `gs://{bucket}/{object}`
    outputs: Map of output name to output info.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class OutputsValue(_messages.Message):
        """Map of output name to output info.

    Messages:
      AdditionalProperty: An additional property for a OutputsValue object.

    Fields:
      additionalProperties: Additional properties of type OutputsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a OutputsValue object.

      Fields:
        key: Name of the additional property.
        value: A TerraformOutput attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('TerraformOutput', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    artifacts = _messages.StringField(1)
    content = _messages.StringField(2)
    outputs = _messages.MessageField('OutputsValue', 3)