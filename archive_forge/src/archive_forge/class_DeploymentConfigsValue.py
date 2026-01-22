from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class DeploymentConfigsValue(_messages.Message):
    """Map of deployment configs to deployments ("admission", "audit",
    "mutation').

    Messages:
      AdditionalProperty: An additional property for a DeploymentConfigsValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        DeploymentConfigsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a DeploymentConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A PolicyControllerPolicyControllerDeploymentConfig attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('PolicyControllerPolicyControllerDeploymentConfig', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)