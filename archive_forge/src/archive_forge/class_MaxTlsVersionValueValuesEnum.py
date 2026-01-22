from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MaxTlsVersionValueValuesEnum(_messages.Enum):
    """Optional. TLS max version used only for Envoy. If not specified, Envoy
    will use default version. Envoy latest: https://www.envoyproxy.io/docs/env
    oy/latest/api-v3/extensions/transport_sockets/tls/v3/common.proto

    Values:
      TLS_VERSION_UNSPECIFIED: <no description>
      TLS_V1_0: <no description>
      TLS_V1_1: <no description>
      TLS_V1_2: <no description>
      TLS_V1_3: <no description>
    """
    TLS_VERSION_UNSPECIFIED = 0
    TLS_V1_0 = 1
    TLS_V1_1 = 2
    TLS_V1_2 = 3
    TLS_V1_3 = 4