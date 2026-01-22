from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkRoutingConfig(_messages.Message):
    """A routing configuration attached to a network resource. The message
  includes the list of routers associated with the network, and a flag
  indicating the type of routing behavior to enforce network-wide.

  Enums:
    RoutingModeValueValuesEnum: The network-wide routing mode to use. If set
      to REGIONAL, this network's Cloud Routers will only advertise routes
      with subnets of this network in the same region as the router. If set to
      GLOBAL, this network's Cloud Routers will advertise routes with all
      subnets of this network, across regions.

  Fields:
    routingMode: The network-wide routing mode to use. If set to REGIONAL,
      this network's Cloud Routers will only advertise routes with subnets of
      this network in the same region as the router. If set to GLOBAL, this
      network's Cloud Routers will advertise routes with all subnets of this
      network, across regions.
  """

    class RoutingModeValueValuesEnum(_messages.Enum):
        """The network-wide routing mode to use. If set to REGIONAL, this
    network's Cloud Routers will only advertise routes with subnets of this
    network in the same region as the router. If set to GLOBAL, this network's
    Cloud Routers will advertise routes with all subnets of this network,
    across regions.

    Values:
      GLOBAL: <no description>
      REGIONAL: <no description>
    """
        GLOBAL = 0
        REGIONAL = 1
    routingMode = _messages.EnumField('RoutingModeValueValuesEnum', 1)