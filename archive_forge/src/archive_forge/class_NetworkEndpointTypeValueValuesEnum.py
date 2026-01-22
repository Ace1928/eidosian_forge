from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkEndpointTypeValueValuesEnum(_messages.Enum):
    """Type of network endpoints in this network endpoint group. Can be one
    of GCE_VM_IP, GCE_VM_IP_PORT, NON_GCP_PRIVATE_IP_PORT, INTERNET_FQDN_PORT,
    INTERNET_IP_PORT, SERVERLESS, PRIVATE_SERVICE_CONNECT, GCE_VM_IP_PORTMAP.

    Values:
      GCE_VM_IP: The network endpoint is represented by an IP address.
      GCE_VM_IP_PORT: The network endpoint is represented by IP address and
        port pair.
      GCE_VM_IP_PORTMAP: The network endpoint is represented by an IP, Port
        and Client Destination Port.
      INTERNET_FQDN_PORT: The network endpoint is represented by fully
        qualified domain name and port.
      INTERNET_IP_PORT: The network endpoint is represented by an internet IP
        address and port.
      NON_GCP_PRIVATE_IP_PORT: The network endpoint is represented by an IP
        address and port. The endpoint belongs to a VM or pod running in a
        customer's on-premises.
      PRIVATE_SERVICE_CONNECT: The network endpoint is either public Google
        APIs or services exposed by other GCP Project with a Service
        Attachment. The connection is set up by private service connect
      SERVERLESS: The network endpoint is handled by specified serverless
        infrastructure.
    """
    GCE_VM_IP = 0
    GCE_VM_IP_PORT = 1
    GCE_VM_IP_PORTMAP = 2
    INTERNET_FQDN_PORT = 3
    INTERNET_IP_PORT = 4
    NON_GCP_PRIVATE_IP_PORT = 5
    PRIVATE_SERVICE_CONNECT = 6
    SERVERLESS = 7