from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExtensionChainExtension(_messages.Message):
    """A single extension in the chain to execute for the matching request.

  Enums:
    SupportedEventsValueListEntryValuesEnum:

  Fields:
    authority: Optional. The `:authority` header in the gRPC request sent from
      Envoy to the extension service. Required for Callout extensions.
    failOpen: Optional. Determines how the proxy behaves if the call to the
      extension fails or times out. When set to `TRUE`, request or response
      processing continues without error. Any subsequent extensions in the
      extension chain are also executed. When set to `FALSE` or the default
      setting of `FALSE` is used, one of the following happens: * If response
      headers have not been delivered to the downstream client, a generic 500
      error is returned to the client. The error response can be tailored by
      configuring a custom error response in the load balancer. * If response
      headers have been delivered, then the HTTP stream to the downstream
      client is reset.
    forwardHeaders: Optional. List of the HTTP headers to forward to the
      extension (from the client or backend). If omitted, all headers are
      sent. Each element is a string indicating the header name.
    name: Required. The name for this extension. The name is logged as part of
      the HTTP request logs. The name must conform with RFC-1034, is
      restricted to lower-cased letters, numbers and hyphens, and can have a
      maximum length of 63 characters. Additionally, the first character must
      be a letter and the last a letter or a number.
    service: Required. The reference to the service that runs the extension.
      Currently only callout extensions are supported here. To configure a
      callout extension, `service` must be a fully-qualified reference to a
      [backend service](https://cloud.google.com/compute/docs/reference/rest/v
      1/backendServices) in the format: `https://www.googleapis.com/compute/v1
      /projects/{project}/regions/{region}/backendServices/{backendService}`
      or `https://www.googleapis.com/compute/v1/projects/{project}/global/back
      endServices/{backendService}`.
    supportedEvents: Optional. A set of events during request or response
      processing for which this extension is called. This field is required
      for the `LbTrafficExtension` resource. It's not relevant for the
      `LbRouteExtension` resource.
    timeout: Optional. Specifies the timeout for each individual message on
      the stream. The timeout must be between 10-1000 milliseconds. Required
      for Callout extensions.
  """

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
    authority = _messages.StringField(1)
    failOpen = _messages.BooleanField(2)
    forwardHeaders = _messages.StringField(3, repeated=True)
    name = _messages.StringField(4)
    service = _messages.StringField(5)
    supportedEvents = _messages.EnumField('SupportedEventsValueListEntryValuesEnum', 6, repeated=True)
    timeout = _messages.StringField(7)