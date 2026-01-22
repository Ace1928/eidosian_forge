from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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