from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpFilter(_messages.Message):
    """HttpFilter is a resource representing http filter definition to be
  programmed in the xDS API compatible clients.

  Messages:
    LabelsValue: Optional. Set of label tags associated with the HttpFilter
      resource.

  Fields:
    config: Required. The configuration needed to enable the HTTP filter. The
      configuration must be JSON formatted and only contain fields defined in
      the protobuf identified in config_type_url.
    configTypeUrl: Required. The fully qualified versioned proto3 type url
      that the filter expects for its configuration. For example:
      'type.googleapis.com/envoy.config.wasm.v2.WasmService'.
    createTime: Output only. The timestamp when the resource was created.
    description: Optional. A free-text description of the resource. Max length
      1024 characters.
    filterName: Required. Name of the HTTP filter defined in the `config`
      field. It is used by the xDS API client to identify specific filter
      implementation the `config` must be applied to. It is different from the
      name of the HttpFilter resource and does not have to be unique. Example:
      'envoy.wasm'.
    labels: Optional. Set of label tags associated with the HttpFilter
      resource.
    name: Required. Name of the HttpFilter resource. It matches pattern
      `projects/*/locations/global/httpFilters/`.
    updateTime: Output only. The timestamp when the resource was updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Set of label tags associated with the HttpFilter resource.

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
    config = _messages.StringField(1)
    configTypeUrl = _messages.StringField(2)
    createTime = _messages.StringField(3)
    description = _messages.StringField(4)
    filterName = _messages.StringField(5)
    labels = _messages.MessageField('LabelsValue', 6)
    name = _messages.StringField(7)
    updateTime = _messages.StringField(8)