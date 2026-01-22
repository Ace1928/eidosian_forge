from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BgpRoute(_messages.Message):
    """A BgpRoute object.

  Enums:
    OriginValueValuesEnum: [Output only] BGP origin (EGP, IGP or INCOMPLETE)

  Fields:
    asPaths: [Output only] AS-PATH for the route
    communities: [Output only] BGP communities in human-readable A:B format.
    destination: [Output only] Destination IP range for the route, in human-
      readable CIDR format
    med: [Output only] BGP multi-exit discriminator
    origin: [Output only] BGP origin (EGP, IGP or INCOMPLETE)
  """

    class OriginValueValuesEnum(_messages.Enum):
        """[Output only] BGP origin (EGP, IGP or INCOMPLETE)

    Values:
      BGP_ORIGIN_EGP: <no description>
      BGP_ORIGIN_IGP: <no description>
      BGP_ORIGIN_INCOMPLETE: <no description>
    """
        BGP_ORIGIN_EGP = 0
        BGP_ORIGIN_IGP = 1
        BGP_ORIGIN_INCOMPLETE = 2
    asPaths = _messages.MessageField('BgpRouteAsPath', 1, repeated=True)
    communities = _messages.StringField(2, repeated=True)
    destination = _messages.MessageField('BgpRouteNetworkLayerReachabilityInformation', 3)
    med = _messages.IntegerField(4, variant=_messages.Variant.UINT32)
    origin = _messages.EnumField('OriginValueValuesEnum', 5)