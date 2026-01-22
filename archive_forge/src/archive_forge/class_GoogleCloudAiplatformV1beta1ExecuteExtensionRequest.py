from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ExecuteExtensionRequest(_messages.Message):
    """Request message for ExtensionExecutionService.ExecuteExtension.

  Messages:
    OperationParamsValue: Optional. Request parameters that will be used for
      executing this operation. The struct should be in a form of map with
      param name as the key and actual param value as the value. E.g. If this
      operation requires a param "name" to be set to "abc". you can set this
      to something like {"name": "abc"}.

  Fields:
    operationId: Required. The desired ID of the operation to be executed in
      this extension as defined in ExtensionOperation.operation_id.
    operationParams: Optional. Request parameters that will be used for
      executing this operation. The struct should be in a form of map with
      param name as the key and actual param value as the value. E.g. If this
      operation requires a param "name" to be set to "abc". you can set this
      to something like {"name": "abc"}.
    runtimeAuthConfig: Optional. Auth config provided at runtime to override
      the default value in Extension.manifest.auth_config. The
      AuthConfig.auth_type should match the value in
      Extension.manifest.auth_config.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class OperationParamsValue(_messages.Message):
        """Optional. Request parameters that will be used for executing this
    operation. The struct should be in a form of map with param name as the
    key and actual param value as the value. E.g. If this operation requires a
    param "name" to be set to "abc". you can set this to something like
    {"name": "abc"}.

    Messages:
      AdditionalProperty: An additional property for a OperationParamsValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a OperationParamsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    operationId = _messages.StringField(1)
    operationParams = _messages.MessageField('OperationParamsValue', 2)
    runtimeAuthConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1AuthConfig', 3)