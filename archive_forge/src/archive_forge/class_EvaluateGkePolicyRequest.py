from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EvaluateGkePolicyRequest(_messages.Message):
    """Request message for PlatformPolicyEvaluationService.EvaluateGkePolicy.

  Enums:
    AttestationModeValueValuesEnum: Optional. Configures the behavior for
      attesting results.

  Messages:
    ResourceValue: Required. JSON or YAML blob representing a Kubernetes
      resource.

  Fields:
    attestationMode: Optional. Configures the behavior for attesting results.
    resource: Required. JSON or YAML blob representing a Kubernetes resource.
  """

    class AttestationModeValueValuesEnum(_messages.Enum):
        """Optional. Configures the behavior for attesting results.

    Values:
      ATTESTATION_MODE_UNSPECIFIED: Unspecified. Results are not attested.
      GENERATE_DEPLOY: Generate and return deploy attestations in DSEE form.
    """
        ATTESTATION_MODE_UNSPECIFIED = 0
        GENERATE_DEPLOY = 1

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ResourceValue(_messages.Message):
        """Required. JSON or YAML blob representing a Kubernetes resource.

    Messages:
      AdditionalProperty: An additional property for a ResourceValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ResourceValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    attestationMode = _messages.EnumField('AttestationModeValueValuesEnum', 1)
    resource = _messages.MessageField('ResourceValue', 2)