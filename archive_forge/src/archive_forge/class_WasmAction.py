from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WasmAction(_messages.Message):
    """`WasmAction` is a resource representing a connection between a
  `WasmPlugin` resource and an `EdgeCacheService` resource. After a
  `WasmAction` resource is created, you can't change its reference to a
  `WasmPlugin` resource.

  Enums:
    SupportedEventsValueListEntryValuesEnum:

  Messages:
    LabelsValue: Optional. Set of label tags associated with the `WasmAction`
      resource.

  Fields:
    createTime: Output only. The timestamp when the resource was created.
    description: Optional. A human-readable description of the resource.
    labels: Optional. Set of label tags associated with the `WasmAction`
      resource.
    name: Required. Name of the `WasmAction` resource in the following format:
      `projects/{project}/locations/{location}/wasmActions/{wasm_action}`.
    supportedEvents: Optional. Determines which of portion of the request /
      response is processed by the plugin. Each value translates to a separate
      plugin invocation. For example, processing request headers involves
      invoking the `ON_HTTP_HEADERS` callback. If empty, both request headers
      and response headers are processed.
    updateTime: Output only. The timestamp when the resource was updated.
    wasmPlugin: Required. The relative resource name of the `WasmPlugin`
      resource to execute in the following format:
      `projects/{project}/locations/{location}/wasmPlugins/{wasm_plugin}`.
  """

    class SupportedEventsValueListEntryValuesEnum(_messages.Enum):
        """SupportedEventsValueListEntryValuesEnum enum type.

    Values:
      EVENT_TYPE_UNSPECIFIED: Unspecified value. Do not use.
      REQUEST_HEADERS: If included in `supported_events`, the HTTP request
        headers are processed.
      RESPONSE_HEADERS: If included in `supported_events`, the HTTP response
        headers are processed.
    """
        EVENT_TYPE_UNSPECIFIED = 0
        REQUEST_HEADERS = 1
        RESPONSE_HEADERS = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Set of label tags associated with the `WasmAction` resource.

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
    description = _messages.StringField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    name = _messages.StringField(4)
    supportedEvents = _messages.EnumField('SupportedEventsValueListEntryValuesEnum', 5, repeated=True)
    updateTime = _messages.StringField(6)
    wasmPlugin = _messages.StringField(7)