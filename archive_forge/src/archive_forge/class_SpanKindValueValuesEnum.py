from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpanKindValueValuesEnum(_messages.Enum):
    """Distinguishes between spans generated in a particular context. For
    example, two spans with the same name may be distinguished using `CLIENT`
    (caller) and `SERVER` (callee) to identify an RPC call.

    Values:
      SPAN_KIND_UNSPECIFIED: Unspecified. Do NOT use as default.
        Implementations MAY assume SpanKind.INTERNAL to be default.
      INTERNAL: Indicates that the span is used internally. Default value.
      SERVER: Indicates that the span covers server-side handling of an RPC or
        other remote network request.
      CLIENT: Indicates that the span covers the client-side wrapper around an
        RPC or other remote request.
      PRODUCER: Indicates that the span describes producer sending a message
        to a broker. Unlike client and server, there is no direct critical
        path latency relationship between producer and consumer spans (e.g.
        publishing a message to a pubsub service).
      CONSUMER: Indicates that the span describes consumer receiving a message
        from a broker. Unlike client and server, there is no direct critical
        path latency relationship between producer and consumer spans (e.g.
        receiving a message from a pubsub service subscription).
    """
    SPAN_KIND_UNSPECIFIED = 0
    INTERNAL = 1
    SERVER = 2
    CLIENT = 3
    PRODUCER = 4
    CONSUMER = 5