from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1Zone(_messages.Message):
    """A zone represents a logical group of related assets within a lake. A
  zone can be used to map to organizational structure or represent stages of
  data readiness from raw to curated. It provides managing behavior that is
  shared or inherited by all contained assets.

  Enums:
    StateValueValuesEnum: Output only. Current state of the zone.
    TypeValueValuesEnum: Required. Immutable. The type of the zone.

  Messages:
    LabelsValue: Optional. User defined labels for the zone.

  Fields:
    assetStatus: Output only. Aggregated status of the underlying assets of
      the zone.
    createTime: Output only. The time when the zone was created.
    description: Optional. Description of the zone.
    discoverySpec: Optional. Specification of the discovery feature applied to
      data in this zone.
    displayName: Optional. User friendly display name.
    labels: Optional. User defined labels for the zone.
    name: Output only. The relative resource name of the zone, of the form: pr
      ojects/{project_number}/locations/{location_id}/lakes/{lake_id}/zones/{z
      one_id}.
    resourceSpec: Required. Specification of the resources that are referenced
      by the assets within this zone.
    state: Output only. Current state of the zone.
    type: Required. Immutable. The type of the zone.
    uid: Output only. System generated globally unique ID for the zone. This
      ID will be different if the zone is deleted and re-created with the same
      name.
    updateTime: Output only. The time when the zone was last updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Current state of the zone.

    Values:
      STATE_UNSPECIFIED: State is not specified.
      ACTIVE: Resource is active, i.e., ready to use.
      CREATING: Resource is under creation.
      DELETING: Resource is under deletion.
      ACTION_REQUIRED: Resource is active but has unresolved actions.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        CREATING = 2
        DELETING = 3
        ACTION_REQUIRED = 4

    class TypeValueValuesEnum(_messages.Enum):
        """Required. Immutable. The type of the zone.

    Values:
      TYPE_UNSPECIFIED: Zone type not specified.
      RAW: A zone that contains data that needs further processing before it
        is considered generally ready for consumption and analytics workloads.
      CURATED: A zone that contains data that is considered to be ready for
        broader consumption and analytics workloads. Curated structured data
        stored in Cloud Storage must conform to certain file formats (parquet,
        avro and orc) and organized in a hive-compatible directory layout.
    """
        TYPE_UNSPECIFIED = 0
        RAW = 1
        CURATED = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. User defined labels for the zone.

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
    assetStatus = _messages.MessageField('GoogleCloudDataplexV1AssetStatus', 1)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    discoverySpec = _messages.MessageField('GoogleCloudDataplexV1ZoneDiscoverySpec', 4)
    displayName = _messages.StringField(5)
    labels = _messages.MessageField('LabelsValue', 6)
    name = _messages.StringField(7)
    resourceSpec = _messages.MessageField('GoogleCloudDataplexV1ZoneResourceSpec', 8)
    state = _messages.EnumField('StateValueValuesEnum', 9)
    type = _messages.EnumField('TypeValueValuesEnum', 10)
    uid = _messages.StringField(11)
    updateTime = _messages.StringField(12)