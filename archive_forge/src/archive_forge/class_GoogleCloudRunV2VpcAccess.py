from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2VpcAccess(_messages.Message):
    """VPC Access settings. For more information on sending traffic to a VPC
  network, visit https://cloud.google.com/run/docs/configuring/connecting-vpc.

  Enums:
    EgressValueValuesEnum: Optional. Traffic VPC egress settings. If not
      provided, it defaults to PRIVATE_RANGES_ONLY.

  Fields:
    connector: VPC Access connector name. Format:
      projects/{project}/locations/{location}/connectors/{connector}, where
      {project} can be project id or number. For more information on sending
      traffic to a VPC network via a connector, visit
      https://cloud.google.com/run/docs/configuring/vpc-connectors.
    egress: Optional. Traffic VPC egress settings. If not provided, it
      defaults to PRIVATE_RANGES_ONLY.
    networkInterfaces: Optional. Direct VPC egress settings. Currently only
      single network interface is supported.
  """

    class EgressValueValuesEnum(_messages.Enum):
        """Optional. Traffic VPC egress settings. If not provided, it defaults to
    PRIVATE_RANGES_ONLY.

    Values:
      VPC_EGRESS_UNSPECIFIED: Unspecified
      ALL_TRAFFIC: All outbound traffic is routed through the VPC connector.
      PRIVATE_RANGES_ONLY: Only private IP ranges are routed through the VPC
        connector.
    """
        VPC_EGRESS_UNSPECIFIED = 0
        ALL_TRAFFIC = 1
        PRIVATE_RANGES_ONLY = 2
    connector = _messages.StringField(1)
    egress = _messages.EnumField('EgressValueValuesEnum', 2)
    networkInterfaces = _messages.MessageField('GoogleCloudRunV2NetworkInterface', 3, repeated=True)