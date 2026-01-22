from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VpcAccessConnector(_messages.Message):
    """VPC access connector specification.

  Enums:
    EgressSettingValueValuesEnum: The egress setting for the connector,
      controlling what traffic is diverted through it.

  Fields:
    egressSetting: The egress setting for the connector, controlling what
      traffic is diverted through it.
    name: Full Serverless VPC Access Connector name e.g. projects/my-
      project/locations/us-central1/connectors/c1.
  """

    class EgressSettingValueValuesEnum(_messages.Enum):
        """The egress setting for the connector, controlling what traffic is
    diverted through it.

    Values:
      EGRESS_SETTING_UNSPECIFIED: <no description>
      ALL_TRAFFIC: Force the use of VPC Access for all egress traffic from the
        function.
      PRIVATE_IP_RANGES: Use the VPC Access Connector for private IP space
        from RFC1918.
    """
        EGRESS_SETTING_UNSPECIFIED = 0
        ALL_TRAFFIC = 1
        PRIVATE_IP_RANGES = 2
    egressSetting = _messages.EnumField('EgressSettingValueValuesEnum', 1)
    name = _messages.StringField(2)