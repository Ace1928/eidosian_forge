from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotificationEndpointGrpcSettings(_messages.Message):
    """Represents a gRPC setting that describes one gRPC notification endpoint
  and the retry duration attempting to send notification to this endpoint.

  Fields:
    authority: Optional. If specified, this field is used to set the authority
      header by the sender of notifications. See
      https://tools.ietf.org/html/rfc7540#section-8.1.2.3
    endpoint: Endpoint to which gRPC notifications are sent. This must be a
      valid gRPCLB DNS name.
    payloadName: Optional. If specified, this field is used to populate the
      "name" field in gRPC requests.
    resendInterval: Optional. This field is used to configure how often to
      send a full update of all non-healthy backends. If unspecified, full
      updates are not sent. If specified, must be in the range between 600
      seconds to 3600 seconds. Nanos are disallowed. Can only be set for
      regional notification endpoints.
    retryDurationSec: How much time (in seconds) is spent attempting
      notification retries until a successful response is received. Default is
      30s. Limit is 20m (1200s). Must be a positive number.
  """
    authority = _messages.StringField(1)
    endpoint = _messages.StringField(2)
    payloadName = _messages.StringField(3)
    resendInterval = _messages.MessageField('Duration', 4)
    retryDurationSec = _messages.IntegerField(5, variant=_messages.Variant.UINT32)