from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AbortInfo(_messages.Message):
    """Details of the final state "abort" and associated resource.

  Enums:
    CauseValueValuesEnum: Causes that the analysis is aborted.

  Fields:
    cause: Causes that the analysis is aborted.
    ipAddress: IP address that caused the abort.
    projectsMissingPermission: List of project IDs the user specified in the
      request but lacks access to. In this case, analysis is aborted with the
      PERMISSION_DENIED cause.
    resourceUri: URI of the resource that caused the abort.
  """

    class CauseValueValuesEnum(_messages.Enum):
        """Causes that the analysis is aborted.

    Values:
      CAUSE_UNSPECIFIED: Cause is unspecified.
      UNKNOWN_NETWORK: Aborted due to unknown network. Deprecated, not used in
        the new tests.
      UNKNOWN_PROJECT: Aborted because no project information can be derived
        from the test input. Deprecated, not used in the new tests.
      NO_EXTERNAL_IP: Aborted because traffic is sent from a public IP to an
        instance without an external IP. Deprecated, not used in the new
        tests.
      UNINTENDED_DESTINATION: Aborted because none of the traces matches
        destination information specified in the input test request.
        Deprecated, not used in the new tests.
      SOURCE_ENDPOINT_NOT_FOUND: Aborted because the source endpoint could not
        be found. Deprecated, not used in the new tests.
      MISMATCHED_SOURCE_NETWORK: Aborted because the source network does not
        match the source endpoint. Deprecated, not used in the new tests.
      DESTINATION_ENDPOINT_NOT_FOUND: Aborted because the destination endpoint
        could not be found. Deprecated, not used in the new tests.
      MISMATCHED_DESTINATION_NETWORK: Aborted because the destination network
        does not match the destination endpoint. Deprecated, not used in the
        new tests.
      UNKNOWN_IP: Aborted because no endpoint with the packet's destination IP
        address is found.
      SOURCE_IP_ADDRESS_NOT_IN_SOURCE_NETWORK: Aborted because the source IP
        address doesn't belong to any of the subnets of the source VPC
        network.
      PERMISSION_DENIED: Aborted because user lacks permission to access all
        or part of the network configurations required to run the test.
      PERMISSION_DENIED_NO_CLOUD_NAT_CONFIGS: Aborted because user lacks
        permission to access Cloud NAT configs required to run the test.
      PERMISSION_DENIED_NO_NEG_ENDPOINT_CONFIGS: Aborted because user lacks
        permission to access Network endpoint group endpoint configs required
        to run the test.
      NO_SOURCE_LOCATION: Aborted because no valid source or destination
        endpoint is derived from the input test request.
      INVALID_ARGUMENT: Aborted because the source or destination endpoint
        specified in the request is invalid. Some examples: - The request
        might contain malformed resource URI, project ID, or IP address. - The
        request might contain inconsistent information (for example, the
        request might include both the instance and the network, but the
        instance might not have a NIC in that network).
      TRACE_TOO_LONG: Aborted because the number of steps in the trace exceeds
        a certain limit. It might be caused by a routing loop.
      INTERNAL_ERROR: Aborted due to internal server error.
      UNSUPPORTED: Aborted because the test scenario is not supported.
      MISMATCHED_IP_VERSION: Aborted because the source and destination
        resources have no common IP version.
      GKE_KONNECTIVITY_PROXY_UNSUPPORTED: Aborted because the connection
        between the control plane and the node of the source cluster is
        initiated by the node and managed by the Konnectivity proxy.
      RESOURCE_CONFIG_NOT_FOUND: Aborted because expected resource
        configuration was missing.
      VM_INSTANCE_CONFIG_NOT_FOUND: Aborted because expected VM instance
        configuration was missing.
      NETWORK_CONFIG_NOT_FOUND: Aborted because expected network configuration
        was missing.
      FIREWALL_CONFIG_NOT_FOUND: Aborted because expected firewall
        configuration was missing.
      ROUTE_CONFIG_NOT_FOUND: Aborted because expected route configuration was
        missing.
      GOOGLE_MANAGED_SERVICE_AMBIGUOUS_PSC_ENDPOINT: Aborted because a PSC
        endpoint selection for the Google-managed service is ambiguous
        (several PSC endpoints satisfy test input).
      SOURCE_PSC_CLOUD_SQL_UNSUPPORTED: Aborted because tests with a PSC-based
        Cloud SQL instance as a source are not supported.
      SOURCE_FORWARDING_RULE_UNSUPPORTED: Aborted because tests with a
        forwarding rule as a source are not supported.
      NON_ROUTABLE_IP_ADDRESS: Aborted because one of the endpoints is a non-
        routable IP address (loopback, link-local, etc).
      UNKNOWN_ISSUE_IN_GOOGLE_MANAGED_PROJECT: Aborted due to an unknown issue
        in the Google-managed project.
      UNSUPPORTED_GOOGLE_MANAGED_PROJECT_CONFIG: Aborted due to an unsupported
        configuration of the Google-managed project.
    """
        CAUSE_UNSPECIFIED = 0
        UNKNOWN_NETWORK = 1
        UNKNOWN_PROJECT = 2
        NO_EXTERNAL_IP = 3
        UNINTENDED_DESTINATION = 4
        SOURCE_ENDPOINT_NOT_FOUND = 5
        MISMATCHED_SOURCE_NETWORK = 6
        DESTINATION_ENDPOINT_NOT_FOUND = 7
        MISMATCHED_DESTINATION_NETWORK = 8
        UNKNOWN_IP = 9
        SOURCE_IP_ADDRESS_NOT_IN_SOURCE_NETWORK = 10
        PERMISSION_DENIED = 11
        PERMISSION_DENIED_NO_CLOUD_NAT_CONFIGS = 12
        PERMISSION_DENIED_NO_NEG_ENDPOINT_CONFIGS = 13
        NO_SOURCE_LOCATION = 14
        INVALID_ARGUMENT = 15
        TRACE_TOO_LONG = 16
        INTERNAL_ERROR = 17
        UNSUPPORTED = 18
        MISMATCHED_IP_VERSION = 19
        GKE_KONNECTIVITY_PROXY_UNSUPPORTED = 20
        RESOURCE_CONFIG_NOT_FOUND = 21
        VM_INSTANCE_CONFIG_NOT_FOUND = 22
        NETWORK_CONFIG_NOT_FOUND = 23
        FIREWALL_CONFIG_NOT_FOUND = 24
        ROUTE_CONFIG_NOT_FOUND = 25
        GOOGLE_MANAGED_SERVICE_AMBIGUOUS_PSC_ENDPOINT = 26
        SOURCE_PSC_CLOUD_SQL_UNSUPPORTED = 27
        SOURCE_FORWARDING_RULE_UNSUPPORTED = 28
        NON_ROUTABLE_IP_ADDRESS = 29
        UNKNOWN_ISSUE_IN_GOOGLE_MANAGED_PROJECT = 30
        UNSUPPORTED_GOOGLE_MANAGED_PROJECT_CONFIG = 31
    cause = _messages.EnumField('CauseValueValuesEnum', 1)
    ipAddress = _messages.StringField(2)
    projectsMissingPermission = _messages.StringField(3, repeated=True)
    resourceUri = _messages.StringField(4)