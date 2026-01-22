from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConvertConfigRequest(_messages.Message):
    """Request message for `ConvertConfig` method.

  Messages:
    ConfigSpecValue: Input configuration For this version of API, the
      supported type is OpenApiSpec

  Fields:
    configSpec: Input configuration For this version of API, the supported
      type is OpenApiSpec
    openApiSpec: The OpenAPI specification for an API.
    serviceName: The service name to use for constructing the normalized
      service configuration equivalent of the provided configuration
      specification.
    swaggerSpec: The swagger specification for an API.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ConfigSpecValue(_messages.Message):
        """Input configuration For this version of API, the supported type is
    OpenApiSpec

    Messages:
      AdditionalProperty: An additional property for a ConfigSpecValue object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ConfigSpecValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    configSpec = _messages.MessageField('ConfigSpecValue', 1)
    openApiSpec = _messages.MessageField('OpenApiSpec', 2)
    serviceName = _messages.StringField(3)
    swaggerSpec = _messages.MessageField('SwaggerSpec', 4)