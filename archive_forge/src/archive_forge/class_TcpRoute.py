from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TcpRoute(_messages.Message):
    """TcpRoute is the resource defining how TCP traffic should be routed by a
  Mesh/Gateway resource.

  Messages:
    LabelsValue: Optional. Set of label tags associated with the TcpRoute
      resource.

  Fields:
    createTime: Output only. The timestamp when the resource was created.
    description: Optional. A free-text description of the resource. Max length
      1024 characters.
    gateways: Optional. Gateways defines a list of gateways this TcpRoute is
      attached to, as one of the routing rules to route the requests served by
      the gateway. Each gateway reference should match the pattern:
      `projects/*/locations/global/gateways/`
    labels: Optional. Set of label tags associated with the TcpRoute resource.
    meshes: Optional. Meshes defines a list of meshes this TcpRoute is
      attached to, as one of the routing rules to route the requests served by
      the mesh. Each mesh reference should match the pattern:
      `projects/*/locations/global/meshes/` The attached Mesh should be of a
      type SIDECAR
    name: Required. Name of the TcpRoute resource. It matches pattern
      `projects/*/locations/global/tcpRoutes/tcp_route_name>`.
    rules: Required. Rules that define how traffic is routed and handled. At
      least one RouteRule must be supplied. If there are multiple rules then
      the action taken will be the first rule to match.
    selfLink: Output only. Server-defined URL of this resource
    updateTime: Output only. The timestamp when the resource was updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Set of label tags associated with the TcpRoute resource.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    gateways = _messages.StringField(3, repeated=True)
    labels = _messages.MessageField('LabelsValue', 4)
    meshes = _messages.StringField(5, repeated=True)
    name = _messages.StringField(6)
    rules = _messages.MessageField('TcpRouteRouteRule', 7, repeated=True)
    selfLink = _messages.StringField(8)
    updateTime = _messages.StringField(9)