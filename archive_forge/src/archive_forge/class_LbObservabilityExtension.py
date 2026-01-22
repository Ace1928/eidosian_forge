from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LbObservabilityExtension(_messages.Message):
    """`LbObservabilityExtension` is a resource that allows to forward traffic
  to a callout backend designed to scan the traffic for security purposes.

  Enums:
    LoadBalancingSchemeValueValuesEnum: Required. All backend services and
      forwarding rules referenced by this extension must share the same load
      balancing scheme. Supported values: `INTERNAL_MANAGED`,
      `EXTERNAL_MANAGED`. For more information, refer to [Choosing a load
      balancer](https://cloud.google.com/load-balancing/docs/backend-service).
    SupportedEventsValueListEntryValuesEnum:

  Messages:
    LabelsValue: Optional. Set of labels associated with the
      `LbObservabilityExtension` resource. The format must comply with [the
      requirements for labels](https://cloud.google.com/compute/docs/labeling-
      resources#requirements) for Google Cloud resources.
    MetadataValue: Optional. The metadata provided here will be included as
      part of the `metadata_context` (of type `google.protobuf.Struct`) in the
      `ProcessingRequest` message sent to the extension server. The metadata
      will be available under the namespace
      `com.google.lb_observability_extension.`. The following variables are
      supported in the metadata Struct: `{forwarding_rule_id}` - substituted
      with the forwarding rule's fully qualified resource name.

  Fields:
    authority: Optional. The `:authority` header in the gRPC request sent from
      Envoy to the extension service.
    createTime: Output only. The timestamp when the resource was created.
    description: Optional. A human-readable description of the resource.
    forwardAttributes: Optional. List of the Envoy attributes to forward to
      the extension server. The attributes provided here will be included as
      part of the `ProcessingRequest.attributes` field (of type `map`), where
      the keys are the attribute names. Refer to the
      [documentation](https://cloud.google.com/service-extensions/docs/cel-
      matcher-language-reference#attributes) for the names of attributes that
      can be forwarded. If omitted, no attributes will be sent. Each element
      is a string indicating the attribute name.
    forwardHeaders: Optional. List of the HTTP headers to forward to the
      extension (from the client or backend). If omitted, all headers are
      sent. Each element is a string indicating the header name.
    forwardingRules: Required. A list of references to the forwarding rules to
      which this service extension is attached to. At least one forwarding
      rule is required.
    labels: Optional. Set of labels associated with the
      `LbObservabilityExtension` resource. The format must comply with [the
      requirements for labels](https://cloud.google.com/compute/docs/labeling-
      resources#requirements) for Google Cloud resources.
    loadBalancingScheme: Required. All backend services and forwarding rules
      referenced by this extension must share the same load balancing scheme.
      Supported values: `INTERNAL_MANAGED`, `EXTERNAL_MANAGED`. For more
      information, refer to [Choosing a load
      balancer](https://cloud.google.com/load-balancing/docs/backend-service).
    metadata: Optional. The metadata provided here will be included as part of
      the `metadata_context` (of type `google.protobuf.Struct`) in the
      `ProcessingRequest` message sent to the extension server. The metadata
      will be available under the namespace
      `com.google.lb_observability_extension.`. The following variables are
      supported in the metadata Struct: `{forwarding_rule_id}` - substituted
      with the forwarding rule's fully qualified resource name.
    name: Required. Identifier. Name of the `LbObservabilityExtension`
      resource in the following format: `projects/{project}/locations/{locatio
      n}/lbObservabilityExtensions/{lb_observability_extension}`.
    service: Required. The reference to the service that runs the extension.
      Must be a reference to a backend service. To configure a Callout
      extension, `service` must be a fully-qualified reference to a [backend s
      ervice](https://cloud.google.com/compute/docs/reference/rest/v1/backendS
      ervices) in the format: `https://www.googleapis.com/compute/v1/projects/
      {project}/regions/{region}/backendServices/{backendService}` or `https:/
      /www.googleapis.com/compute/v1/projects/{project}/global/backendServices
      /{backendService}`.
    supportedEvents: Optional. A set of events during request or response
      processing for which this extension is called.
    updateTime: Output only. The timestamp when the resource was updated.
  """

    class LoadBalancingSchemeValueValuesEnum(_messages.Enum):
        """Required. All backend services and forwarding rules referenced by this
    extension must share the same load balancing scheme. Supported values:
    `INTERNAL_MANAGED`, `EXTERNAL_MANAGED`. For more information, refer to
    [Choosing a load balancer](https://cloud.google.com/load-
    balancing/docs/backend-service).

    Values:
      LOAD_BALANCING_SCHEME_UNSPECIFIED: Default value. Do not use.
      INTERNAL_MANAGED: Signifies that this is used for Internal HTTP(S) Load
        Balancing.
      EXTERNAL_MANAGED: Signifies that this is used for External Managed
        HTTP(S) Load Balancing.
    """
        LOAD_BALANCING_SCHEME_UNSPECIFIED = 0
        INTERNAL_MANAGED = 1
        EXTERNAL_MANAGED = 2

    class SupportedEventsValueListEntryValuesEnum(_messages.Enum):
        """SupportedEventsValueListEntryValuesEnum enum type.

    Values:
      EVENT_TYPE_UNSPECIFIED: Unspecified value. Do not use.
      REQUEST_HEADERS: If included in `supported_events`, the extension is
        called when the HTTP request headers arrive.
      REQUEST_BODY: If included in `supported_events`, the extension is called
        when the HTTP request body arrives.
      RESPONSE_HEADERS: If included in `supported_events`, the extension is
        called when the HTTP response headers arrive.
      RESPONSE_BODY: If included in `supported_events`, the extension is
        called when the HTTP response body arrives.
      REQUEST_TRAILERS: If included in `supported_events`, the extension is
        called when the HTTP request trailers arrives.
      RESPONSE_TRAILERS: If included in `supported_events`, the extension is
        called when the HTTP response trailers arrives.
    """
        EVENT_TYPE_UNSPECIFIED = 0
        REQUEST_HEADERS = 1
        REQUEST_BODY = 2
        RESPONSE_HEADERS = 3
        RESPONSE_BODY = 4
        REQUEST_TRAILERS = 5
        RESPONSE_TRAILERS = 6

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Set of labels associated with the `LbObservabilityExtension`
    resource. The format must comply with [the requirements for
    labels](https://cloud.google.com/compute/docs/labeling-
    resources#requirements) for Google Cloud resources.

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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetadataValue(_messages.Message):
        """Optional. The metadata provided here will be included as part of the
    `metadata_context` (of type `google.protobuf.Struct`) in the
    `ProcessingRequest` message sent to the extension server. The metadata
    will be available under the namespace
    `com.google.lb_observability_extension.`. The following variables are
    supported in the metadata Struct: `{forwarding_rule_id}` - substituted
    with the forwarding rule's fully qualified resource name.

    Messages:
      AdditionalProperty: An additional property for a MetadataValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    authority = _messages.StringField(1)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    forwardAttributes = _messages.StringField(4, repeated=True)
    forwardHeaders = _messages.StringField(5, repeated=True)
    forwardingRules = _messages.StringField(6, repeated=True)
    labels = _messages.MessageField('LabelsValue', 7)
    loadBalancingScheme = _messages.EnumField('LoadBalancingSchemeValueValuesEnum', 8)
    metadata = _messages.MessageField('MetadataValue', 9)
    name = _messages.StringField(10)
    service = _messages.StringField(11)
    supportedEvents = _messages.EnumField('SupportedEventsValueListEntryValuesEnum', 12, repeated=True)
    updateTime = _messages.StringField(13)