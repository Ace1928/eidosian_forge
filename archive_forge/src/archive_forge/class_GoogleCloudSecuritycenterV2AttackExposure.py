from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2AttackExposure(_messages.Message):
    """An attack exposure contains the results of an attack path simulation
  run.

  Enums:
    StateValueValuesEnum: Output only. What state this AttackExposure is in.
      This captures whether or not an attack exposure has been calculated or
      not.

  Fields:
    attackExposureResult: The resource name of the attack path simulation
      result that contains the details regarding this attack exposure score.
      Example: organizations/123/simulations/456/attackExposureResults/789
    exposedHighValueResourcesCount: The number of high value resources that
      are exposed as a result of this finding.
    exposedLowValueResourcesCount: The number of high value resources that are
      exposed as a result of this finding.
    exposedMediumValueResourcesCount: The number of medium value resources
      that are exposed as a result of this finding.
    latestCalculationTime: The most recent time the attack exposure was
      updated on this finding.
    score: A number between 0 (inclusive) and infinity that represents how
      important this finding is to remediate. The higher the score, the more
      important it is to remediate.
    state: Output only. What state this AttackExposure is in. This captures
      whether or not an attack exposure has been calculated or not.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. What state this AttackExposure is in. This captures
    whether or not an attack exposure has been calculated or not.

    Values:
      STATE_UNSPECIFIED: The state is not specified.
      CALCULATED: The attack exposure has been calculated.
      NOT_CALCULATED: The attack exposure has not been calculated.
    """
        STATE_UNSPECIFIED = 0
        CALCULATED = 1
        NOT_CALCULATED = 2
    attackExposureResult = _messages.StringField(1)
    exposedHighValueResourcesCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    exposedLowValueResourcesCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    exposedMediumValueResourcesCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    latestCalculationTime = _messages.StringField(5)
    score = _messages.FloatField(6)
    state = _messages.EnumField('StateValueValuesEnum', 7)