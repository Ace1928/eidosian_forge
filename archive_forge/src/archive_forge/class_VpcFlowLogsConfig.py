from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VpcFlowLogsConfig(_messages.Message):
    """A configuration to generate VPC Flow Logs.

  Enums:
    AggregationIntervalValueValuesEnum: Optional. The aggregation interval for
      the logs. Default value is INTERVAL_5_SEC.
    MetadataValueValuesEnum: Optional. Configures whether all, none or a
      subset of metadata fields should be added to the reported VPC flow logs.
      Default value is INCLUDE_ALL_METADATA.
    StateValueValuesEnum: Optional. The state of the VPC Flow Log. Default
      value is ENABLED.

  Fields:
    aggregationInterval: Optional. The aggregation interval for the logs.
      Default value is INTERVAL_5_SEC.
    createTime: Output only. The time the config was created.
    description: Optional. The user-supplied description of the VPC Flow Logs
      configuration. Maximum of 512 characters.
    filterExpr: Export filter used to define which VPC flow logs should be
      logged.
    flowSampling: Optional. The value of the field must be in [0, 1]. The
      sampling rate of VPC flow logs within the subnetwork where 1.0 means all
      collected logs are reported and 0.0 means no logs are reported. Default
      value is 1.0.
    interconnectAttachment: Traffic will be logged from the Interconnect
      Attachment. Format:
      projects/{project_id}/locations/{region}/interconnectAttachments/{name}
    metadata: Optional. Configures whether all, none or a subset of metadata
      fields should be added to the reported VPC flow logs. Default value is
      INCLUDE_ALL_METADATA.
    metadataFields: Optional. Custom metadata fields to include in the
      reported VPC flow logs. Can only be specified if "metadata" was set to
      CUSTOM_METADATA.
    name: Identifier. Unique name of the configuration using the form:
      `projects/{project_id}/locations/global/vpcFlowLogs/{vpc_flow_log}`
    network: Traffic will be logged from VMs, VPN tunnels and Interconnect
      Attachments within the network. Format:
      projects/{project_id}/networks/{name}
    state: Optional. The state of the VPC Flow Log. Default value is ENABLED.
    subnet: Traffic will be logged from VMs within the subnetwork. Format:
      projects/{project_id}/locations/{region}/subnetworks/{name}
    updateTime: Output only. The time the config was updated.
    vpnTunnel: Traffic will be logged from the VPN Tunnel. Format:
      projects/{project_id}/locations/{region}/vpnTunnels/{name}
  """

    class AggregationIntervalValueValuesEnum(_messages.Enum):
        """Optional. The aggregation interval for the logs. Default value is
    INTERVAL_5_SEC.

    Values:
      AGGREGATION_INTERVAL_UNSPECIFIED: If not specified, will default to
        INTERVAL_5_SEC.
      INTERVAL_5_SEC: Aggregate logs in 5s intervals.
      INTERVAL_30_SEC: Aggregate logs in 30s intervals.
      INTERVAL_1_MIN: Aggregate logs in 1m intervals.
      INTERVAL_5_MIN: Aggregate logs in 5m intervals.
      INTERVAL_10_MIN: Aggregate logs in 10m intervals.
      INTERVAL_15_MIN: Aggregate logs in 15m intervals.
    """
        AGGREGATION_INTERVAL_UNSPECIFIED = 0
        INTERVAL_5_SEC = 1
        INTERVAL_30_SEC = 2
        INTERVAL_1_MIN = 3
        INTERVAL_5_MIN = 4
        INTERVAL_10_MIN = 5
        INTERVAL_15_MIN = 6

    class MetadataValueValuesEnum(_messages.Enum):
        """Optional. Configures whether all, none or a subset of metadata fields
    should be added to the reported VPC flow logs. Default value is
    INCLUDE_ALL_METADATA.

    Values:
      METADATA_UNSPECIFIED: If not specified, will default to
        INCLUDE_ALL_METADATA.
      INCLUDE_ALL_METADATA: Include all metadata fields.
      EXCLUDE_ALL_METADATA: Exclude all metadata fields.
      CUSTOM_METADATA: Include only custom fields (specified in
        metadata_fields).
    """
        METADATA_UNSPECIFIED = 0
        INCLUDE_ALL_METADATA = 1
        EXCLUDE_ALL_METADATA = 2
        CUSTOM_METADATA = 3

    class StateValueValuesEnum(_messages.Enum):
        """Optional. The state of the VPC Flow Log. Default value is ENABLED.

    Values:
      STATE_UNSPECIFIED: If not specified, will default to ENABLED.
      ENABLED: When ENABLED, this configuration will generate logs.
      DISABLED: When DISABLED, this configuration will not generate logs.
    """
        STATE_UNSPECIFIED = 0
        ENABLED = 1
        DISABLED = 2
    aggregationInterval = _messages.EnumField('AggregationIntervalValueValuesEnum', 1)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    filterExpr = _messages.StringField(4)
    flowSampling = _messages.FloatField(5, variant=_messages.Variant.FLOAT)
    interconnectAttachment = _messages.StringField(6)
    metadata = _messages.EnumField('MetadataValueValuesEnum', 7)
    metadataFields = _messages.StringField(8, repeated=True)
    name = _messages.StringField(9)
    network = _messages.StringField(10)
    state = _messages.EnumField('StateValueValuesEnum', 11)
    subnet = _messages.StringField(12)
    updateTime = _messages.StringField(13)
    vpnTunnel = _messages.StringField(14)