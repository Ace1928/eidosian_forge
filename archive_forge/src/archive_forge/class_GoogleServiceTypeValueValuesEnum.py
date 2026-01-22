from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleServiceTypeValueValuesEnum(_messages.Enum):
    """Recognized type of a Google Service.

    Values:
      GOOGLE_SERVICE_TYPE_UNSPECIFIED: Unspecified Google Service.
      IAP: Identity aware proxy. https://cloud.google.com/iap/docs/using-tcp-
        forwarding
      GFE_PROXY_OR_HEALTH_CHECK_PROBER: One of two services sharing IP ranges:
        * Load Balancer proxy * Centralized Health Check prober
        https://cloud.google.com/load-balancing/docs/firewall-rules
      CLOUD_DNS: Connectivity from Cloud DNS to forwarding targets or
        alternate name servers that use private routing.
        https://cloud.google.com/dns/docs/zones/forwarding-zones#firewall-
        rules https://cloud.google.com/dns/docs/policies#firewall-rules
      GOOGLE_API: private.googleapis.com and restricted.googleapis.com
      GOOGLE_API_PSC: Google API via Private Service Connect.
        https://cloud.google.com/vpc/docs/configure-private-service-connect-
        apis
      GOOGLE_API_VPC_SC: Google API via VPC Service Controls.
        https://cloud.google.com/vpc/docs/configure-private-service-connect-
        apis
    """
    GOOGLE_SERVICE_TYPE_UNSPECIFIED = 0
    IAP = 1
    GFE_PROXY_OR_HEALTH_CHECK_PROBER = 2
    CLOUD_DNS = 3
    GOOGLE_API = 4
    GOOGLE_API_PSC = 5
    GOOGLE_API_VPC_SC = 6