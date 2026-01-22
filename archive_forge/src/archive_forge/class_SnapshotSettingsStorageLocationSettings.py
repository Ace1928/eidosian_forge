from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SnapshotSettingsStorageLocationSettings(_messages.Message):
    """A SnapshotSettingsStorageLocationSettings object.

  Enums:
    PolicyValueValuesEnum: The chosen location policy.

  Messages:
    LocationsValue: When the policy is SPECIFIC_LOCATIONS, snapshots will be
      stored in the locations listed in this field. Keys are GCS bucket
      locations.

  Fields:
    locations: When the policy is SPECIFIC_LOCATIONS, snapshots will be stored
      in the locations listed in this field. Keys are GCS bucket locations.
    policy: The chosen location policy.
  """

    class PolicyValueValuesEnum(_messages.Enum):
        """The chosen location policy.

    Values:
      LOCAL_REGION: Store snapshot in the same region as with the originating
        disk. No additional parameters are needed.
      NEAREST_MULTI_REGION: Store snapshot to the nearest multi region GCS
        bucket, relative to the originating disk. No additional parameters are
        needed.
      SPECIFIC_LOCATIONS: Store snapshot in the specific locations, as
        specified by the user. The list of regions to store must be defined
        under the `locations` field.
      STORAGE_LOCATION_POLICY_UNSPECIFIED: <no description>
    """
        LOCAL_REGION = 0
        NEAREST_MULTI_REGION = 1
        SPECIFIC_LOCATIONS = 2
        STORAGE_LOCATION_POLICY_UNSPECIFIED = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LocationsValue(_messages.Message):
        """When the policy is SPECIFIC_LOCATIONS, snapshots will be stored in the
    locations listed in this field. Keys are GCS bucket locations.

    Messages:
      AdditionalProperty: An additional property for a LocationsValue object.

    Fields:
      additionalProperties: Additional properties of type LocationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LocationsValue object.

      Fields:
        key: Name of the additional property.
        value: A
          SnapshotSettingsStorageLocationSettingsStorageLocationPreference
          attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('SnapshotSettingsStorageLocationSettingsStorageLocationPreference', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    locations = _messages.MessageField('LocationsValue', 1)
    policy = _messages.EnumField('PolicyValueValuesEnum', 2)