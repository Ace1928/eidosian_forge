from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LbTrafficExtension(_messages.Message):
    """`LbTrafficExtension` is a resource that lets the extension service
  modify the headers and payloads of both requests and responses without
  impacting the choice of backend services or any other security policies
  associated with the backend service.

  Enums:
    LoadBalancingSchemeValueValuesEnum: Required. All backend services and
      forwarding rules referenced by this extension must share the same load
      balancing scheme. Supported values: `INTERNAL_MANAGED`,
      `EXTERNAL_MANAGED`. For more information, refer to [Choosing a load
      balancer](https://cloud.google.com/load-balancing/docs/backend-service).

  Messages:
    LabelsValue: Optional. Set of labels associated with the
      `LbTrafficExtension` resource. The format must comply with [the
      requirements for labels](https://cloud.google.com/compute/docs/labeling-
      resources#requirements) for Google Cloud resources.

  Fields:
    createTime: Output only. The timestamp when the resource was created.
    description: Optional. A human-readable description of the resource.
    extensionChains: Required. A set of ordered extension chains that contain
      the match conditions and extensions to execute. Match conditions for
      each extension chain are evaluated in sequence for a given request. The
      first extension chain that has a condition that matches the request is
      executed. Any subsequent extension chains do not execute. Limited to 5
      extension chains per resource.
    forwardingRules: Required. A list of references to the forwarding rules to
      which this service extension is attached to. At least one forwarding
      rule is required. There can be only one `LBTrafficExtension` resource
      per forwarding rule.
    labels: Optional. Set of labels associated with the `LbTrafficExtension`
      resource. The format must comply with [the requirements for
      labels](https://cloud.google.com/compute/docs/labeling-
      resources#requirements) for Google Cloud resources.
    loadBalancingScheme: Required. All backend services and forwarding rules
      referenced by this extension must share the same load balancing scheme.
      Supported values: `INTERNAL_MANAGED`, `EXTERNAL_MANAGED`. For more
      information, refer to [Choosing a load
      balancer](https://cloud.google.com/load-balancing/docs/backend-service).
    name: Required. Identifier. Name of the `LbTrafficExtension` resource in
      the following format: `projects/{project}/locations/{location}/lbTraffic
      Extensions/{lb_traffic_extension}`.
    updateTime: Output only. The timestamp when the resource was updated.
  """

    class LoadBalancingSchemeValueValuesEnum(_messages.Enum):
        """Required. All backend services and forwarding rules referenced by this
    extension must share the same load balancing scheme. Supported values:
    `INTERNAL_MANAGED`, `EXTERNAL_MANAGED`. For more information, refer to
    [Choosing a load balancer](https://cloud.google.com/load-
    balancing/docs/backend-service).

    Values:
      LOAD_BALANCING_SCHEME_UNSPECIFIED: Default value. Do not use.
      INTERNAL_MANAGED: Signifies that this is used for Internal HTTP(S) Load
        Balancing.
      EXTERNAL_MANAGED: Signifies that this is used for External Managed
        HTTP(S) Load Balancing.
    """
        LOAD_BALANCING_SCHEME_UNSPECIFIED = 0
        INTERNAL_MANAGED = 1
        EXTERNAL_MANAGED = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Set of labels associated with the `LbTrafficExtension`
    resource. The format must comply with [the requirements for
    labels](https://cloud.google.com/compute/docs/labeling-
    resources#requirements) for Google Cloud resources.

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
    extensionChains = _messages.MessageField('ExtensionChain', 3, repeated=True)
    forwardingRules = _messages.StringField(4, repeated=True)
    labels = _messages.MessageField('LabelsValue', 5)
    loadBalancingScheme = _messages.EnumField('LoadBalancingSchemeValueValuesEnum', 6)
    name = _messages.StringField(7)
    updateTime = _messages.StringField(8)