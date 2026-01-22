from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalInstallationParams(_messages.Message):
    """Information about the device installation parameters.

  Enums:
    HeightTypeValueValuesEnum: Specifies how the height is measured.

  Fields:
    antennaAzimuth: Boresight direction of the horizontal plane of the antenna
      in degrees with respect to true north. The value of this parameter is an
      integer with a value between 0 and 359 inclusive. A value of 0 degrees
      means true north; a value of 90 degrees means east. This parameter is
      optional for Category A devices and conditional for Category B devices.
    antennaBeamwidth: 3-dB antenna beamwidth of the antenna in the horizontal-
      plane in degrees. This parameter is an unsigned integer having a value
      between 0 and 360 (degrees) inclusive; it is optional for Category A
      devices and conditional for Category B devices.
    antennaDowntilt: Antenna downtilt in degrees and is an integer with a
      value between -90 and +90 inclusive; a negative value means the antenna
      is tilted up (above horizontal). This parameter is optional for Category
      A devices and conditional for Category B devices.
    antennaGain: Peak antenna gain in dBi. This parameter is an integer with a
      value between -127 and +128 (dBi) inclusive.
    antennaModel: If an external antenna is used, the antenna model is
      optionally provided in this field. The string has a maximum length of
      128 octets.
    cpeCbsdIndication: If present, this parameter specifies whether the CBSD
      is a CPE-CBSD or not.
    eirpCapability: This parameter is the maximum device EIRP in units of
      dBm/10MHz and is an integer with a value between -127 and +47 (dBm/10
      MHz) inclusive. If not included, SAS interprets it as maximum allowable
      EIRP in units of dBm/10MHz for device category.
    height: Device antenna height in meters. When the `heightType` parameter
      value is "AGL", the antenna height should be given relative to ground
      level. When the `heightType` parameter value is "AMSL", it is given with
      respect to WGS84 datum.
    heightType: Specifies how the height is measured.
    horizontalAccuracy: A positive number in meters to indicate accuracy of
      the device antenna horizontal location. This optional parameter should
      only be present if its value is less than the FCC requirement of 50
      meters.
    indoorDeployment: Whether the device antenna is indoor or not. `true`:
      indoor. `false`: outdoor.
    latitude: Latitude of the device antenna location in degrees relative to
      the WGS 84 datum. The allowed range is from -90.000000 to +90.000000.
      Positive values represent latitudes north of the equator; negative
      values south of the equator.
    longitude: Longitude of the device antenna location in degrees relative to
      the WGS 84 datum. The allowed range is from -180.000000 to +180.000000.
      Positive values represent longitudes east of the prime meridian;
      negative values west of the prime meridian.
    verticalAccuracy: A positive number in meters to indicate accuracy of the
      device antenna vertical location. This optional parameter should only be
      present if its value is less than the FCC requirement of 3 meters.
  """

    class HeightTypeValueValuesEnum(_messages.Enum):
        """Specifies how the height is measured.

    Values:
      HEIGHT_TYPE_UNSPECIFIED: Unspecified height type.
      HEIGHT_TYPE_AGL: AGL height is measured relative to the ground level.
      HEIGHT_TYPE_AMSL: AMSL height is measured relative to the mean sea
        level.
    """
        HEIGHT_TYPE_UNSPECIFIED = 0
        HEIGHT_TYPE_AGL = 1
        HEIGHT_TYPE_AMSL = 2
    antennaAzimuth = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    antennaBeamwidth = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    antennaDowntilt = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    antennaGain = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    antennaModel = _messages.StringField(5)
    cpeCbsdIndication = _messages.BooleanField(6)
    eirpCapability = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    height = _messages.FloatField(8)
    heightType = _messages.EnumField('HeightTypeValueValuesEnum', 9)
    horizontalAccuracy = _messages.FloatField(10)
    indoorDeployment = _messages.BooleanField(11)
    latitude = _messages.FloatField(12)
    longitude = _messages.FloatField(13)
    verticalAccuracy = _messages.FloatField(14)