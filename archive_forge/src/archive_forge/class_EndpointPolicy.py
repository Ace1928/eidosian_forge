from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EndpointPolicy(_messages.Message):
    """EndpointPolicy is a resource that helps apply desired configuration on
  the endpoints that match specific criteria. For example, this resource can
  be used to apply "authentication config" an all endpoints that serve on port
  8080.

  Enums:
    TypeValueValuesEnum: Required. The type of endpoint policy. This is
      primarily used to validate the configuration.

  Messages:
    LabelsValue: Optional. Set of label tags associated with the
      EndpointPolicy resource.

  Fields:
    authorizationPolicy: Optional. This field specifies the URL of
      AuthorizationPolicy resource that applies authorization policies to the
      inbound traffic at the matched endpoints. Refer to Authorization. If
      this field is not specified, authorization is disabled(no authz checks)
      for this endpoint.
    clientTlsPolicy: Optional. A URL referring to a ClientTlsPolicy resource.
      ClientTlsPolicy can be set to specify the authentication for traffic
      from the proxy to the actual endpoints. More specifically, it is applied
      to the outgoing traffic from the proxy to the endpoint. This is
      typically used for sidecar model where the proxy identifies itself as
      endpoint to the control plane, with the connection between sidecar and
      endpoint requiring authentication. If this field is not set,
      authentication is disabled(open). Applicable only when
      EndpointPolicyType is SIDECAR_PROXY.
    createTime: Output only. The timestamp when the resource was created.
    description: Optional. A free-text description of the resource. Max length
      1024 characters.
    endpointMatcher: Required. A matcher that selects endpoints to which the
      policies should be applied.
    labels: Optional. Set of label tags associated with the EndpointPolicy
      resource.
    name: Required. Name of the EndpointPolicy resource. It matches pattern
      `projects/{project}/locations/global/endpointPolicies/{endpoint_policy}`
      .
    serverTlsPolicy: Optional. A URL referring to ServerTlsPolicy resource.
      ServerTlsPolicy is used to determine the authentication policy to be
      applied to terminate the inbound traffic at the identified backends. If
      this field is not set, authentication is disabled(open) for this
      endpoint.
    trafficPortSelector: Optional. Port selector for the (matched) endpoints.
      If no port selector is provided, the matched config is applied to all
      ports.
    type: Required. The type of endpoint policy. This is primarily used to
      validate the configuration.
    updateTime: Output only. The timestamp when the resource was updated.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Required. The type of endpoint policy. This is primarily used to
    validate the configuration.

    Values:
      ENDPOINT_POLICY_TYPE_UNSPECIFIED: Default value. Must not be used.
      SIDECAR_PROXY: Represents a proxy deployed as a sidecar.
      GRPC_SERVER: Represents a proxyless gRPC backend.
    """
        ENDPOINT_POLICY_TYPE_UNSPECIFIED = 0
        SIDECAR_PROXY = 1
        GRPC_SERVER = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Set of label tags associated with the EndpointPolicy
    resource.

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
    authorizationPolicy = _messages.StringField(1)
    clientTlsPolicy = _messages.StringField(2)
    createTime = _messages.StringField(3)
    description = _messages.StringField(4)
    endpointMatcher = _messages.MessageField('EndpointMatcher', 5)
    labels = _messages.MessageField('LabelsValue', 6)
    name = _messages.StringField(7)
    serverTlsPolicy = _messages.StringField(8)
    trafficPortSelector = _messages.MessageField('TrafficPortSelector', 9)
    type = _messages.EnumField('TypeValueValuesEnum', 10)
    updateTime = _messages.StringField(11)