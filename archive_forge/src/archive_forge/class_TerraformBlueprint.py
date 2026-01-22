from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TerraformBlueprint(_messages.Message):
    """TerraformBlueprint describes the source of a Terraform root module which
  describes the resources and configs to be deployed.

  Messages:
    InputValuesValue: Input variable values for the Terraform blueprint.

  Fields:
    gcsSource: Required. URI of an object in Google Cloud Storage. Format:
      `gs://{bucket}/{object}` URI may also specify an object version for
      zipped objects. Format: `gs://{bucket}/{object}#{version}`
    gitSource: Required. URI of a public Git repo.
    inputValues: Input variable values for the Terraform blueprint.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class InputValuesValue(_messages.Message):
        """Input variable values for the Terraform blueprint.

    Messages:
      AdditionalProperty: An additional property for a InputValuesValue
        object.

    Fields:
      additionalProperties: Additional properties of type InputValuesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a InputValuesValue object.

      Fields:
        key: Name of the additional property.
        value: A TerraformVariable attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('TerraformVariable', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    gcsSource = _messages.StringField(1)
    gitSource = _messages.MessageField('GitSource', 2)
    inputValues = _messages.MessageField('InputValuesValue', 3)