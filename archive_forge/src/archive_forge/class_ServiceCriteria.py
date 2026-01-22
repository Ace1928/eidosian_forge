from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceCriteria(_messages.Message):
    """Criteria to apply to identify components belonging to this service.

  Enums:
    KeyValueValuesEnum: Required. Key for criteria.

  Fields:
    key: Required. Key for criteria.
    value: Required. Criteria value to match against for the associated
      criteria key. Example: //compute.googleapis.com/projects/123/regions/us-
      west1/forwardingRules/fr1
  """

    class KeyValueValuesEnum(_messages.Enum):
        """Required. Key for criteria.

    Values:
      CRITERIA_KEY_UNSPECIFIED: Default. Criteria.key is unspecified.
      FORWARDING_RULE: Criteria type of Forwarding Rule. Example value:
        //compute.googleapis.com/projects/123/regions/us-
        west1/forwardingRules/fr1
      GKE_GATEWAY: Criteria type of GKE Gateway. Example value:
        //container.googleapis.com/projects/123/zones/us-
        central1-a/clusters/my-cluster/k8s/apis/gateway.networking.k8s.io/v1al
        pha2/namespaces/default/gateways/my-gateway
      APP_HUB_SERVICE: Criteria type of App Hub service. Example value:
        //servicedirectory.googleapis.com/projects/1234/locations/us-
        west1/namespaces/my-ns/services/gshoe-service
      APP_HUB_WORKLOAD: Criteria type of App Hub workload. Example value:
        //servicedirectory.googleapis.com/projects/1234/locations/us-
        west1/namespaces/my-ns/workloads/gshoe-workload
    """
        CRITERIA_KEY_UNSPECIFIED = 0
        FORWARDING_RULE = 1
        GKE_GATEWAY = 2
        APP_HUB_SERVICE = 3
        APP_HUB_WORKLOAD = 4
    key = _messages.EnumField('KeyValueValuesEnum', 1)
    value = _messages.StringField(2)