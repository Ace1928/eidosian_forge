from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IngressValueValuesEnum(_messages.Enum):
    """Optional. Provides the ingress settings for this Service. On output,
    returns the currently observed ingress settings, or
    INGRESS_TRAFFIC_UNSPECIFIED if no revision is active.

    Values:
      INGRESS_TRAFFIC_UNSPECIFIED: Unspecified
      INGRESS_TRAFFIC_ALL: All inbound traffic is allowed.
      INGRESS_TRAFFIC_INTERNAL_ONLY: Only internal traffic is allowed.
      INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER: Both internal and Google Cloud
        Load Balancer traffic is allowed.
      INGRESS_TRAFFIC_NONE: No ingress traffic is allowed.
    """
    INGRESS_TRAFFIC_UNSPECIFIED = 0
    INGRESS_TRAFFIC_ALL = 1
    INGRESS_TRAFFIC_INTERNAL_ONLY = 2
    INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER = 3
    INGRESS_TRAFFIC_NONE = 4