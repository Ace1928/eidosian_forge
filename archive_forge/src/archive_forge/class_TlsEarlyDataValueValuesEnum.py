from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TlsEarlyDataValueValuesEnum(_messages.Enum):
    """ Specifies whether TLS 1.3 0-RTT Data ("Early Data") should be
    accepted for this service. Early Data allows a TLS resumption handshake to
    include the initial application payload (a HTTP request) alongside the
    handshake, reducing the effective round trips to "zero". This applies to
    TLS 1.3 connections over TCP (HTTP/2) as well as over UDP (QUIC/h3). This
    can improve application performance, especially on networks where
    interruptions may be common, such as on mobile. Requests with Early Data
    will have the "Early-Data" HTTP header set on the request, with a value of
    "1", to allow the backend to determine whether Early Data was included.
    Note: TLS Early Data may allow requests to be replayed, as the data is
    sent to the backend before the handshake has fully completed. Applications
    that allow idempotent HTTP methods to make non-idempotent changes, such as
    a GET request updating a database, should not accept Early Data on those
    requests, and reject requests with the "Early-Data: 1" HTTP header by
    returning a HTTP 425 (Too Early) status code, in order to remain RFC
    compliant. The default value is DISABLED.

    Values:
      DISABLED: TLS 1.3 Early Data is not advertised, and any (invalid)
        attempts to send Early Data will be rejected by closing the
        connection.
      PERMISSIVE: This enables TLS 1.3 0-RTT, and only allows Early Data to be
        included on requests with safe HTTP methods (GET, HEAD, OPTIONS,
        TRACE). This mode does not enforce any other limitations for requests
        with Early Data. The application owner should validate that Early Data
        is acceptable for a given request path.
      STRICT: This enables TLS 1.3 0-RTT, and only allows Early Data to be
        included on requests with safe HTTP methods (GET, HEAD, OPTIONS,
        TRACE) without query parameters. Requests that send Early Data with
        non-idempotent HTTP methods or with query parameters will be rejected
        with a HTTP 425.
    """
    DISABLED = 0
    PERMISSIVE = 1
    STRICT = 2