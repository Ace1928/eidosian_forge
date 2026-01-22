from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsNetworkPeeringsPeeringRoutesListRequest(_messages.Message):
    """A VmwareengineProjectsLocationsNetworkPeeringsPeeringRoutesListRequest
  object.

  Fields:
    filter: A filter expression that matches resources returned in the
      response. Currently, only filtering on the `direction` field is
      supported. To return routes imported from the peer network, provide
      "direction=INCOMING". To return routes exported from the VMware Engine
      network, provide "direction=OUTGOING". Other filter expressions return
      an error.
    pageSize: The maximum number of peering routes to return in one page. The
      service may return fewer than this value. The maximum value is coerced
      to 1000. The default value of this field is 500.
    pageToken: A page token, received from a previous `ListPeeringRoutes`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to `ListPeeringRoutes` must match the call
      that provided the page token.
    parent: Required. The resource name of the network peering to retrieve
      peering routes from. Resource names are schemeless URIs that follow the
      conventions in https://cloud.google.com/apis/design/resource_names. For
      example: `projects/my-project/locations/global/networkPeerings/my-
      peering`
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)