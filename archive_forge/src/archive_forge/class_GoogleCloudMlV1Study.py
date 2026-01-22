from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1Study(_messages.Message):
    """A message representing a Study.

  Enums:
    StateValueValuesEnum: Output only. The detailed state of a study.

  Fields:
    createTime: Output only. Time at which the study was created.
    inactiveReason: Output only. A human readable reason why the Study is
      inactive. This should be empty if a study is ACTIVE or COMPLETED.
    name: Output only. The name of a study.
    state: Output only. The detailed state of a study.
    studyConfig: Required. Configuration of the study.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The detailed state of a study.

    Values:
      STATE_UNSPECIFIED: The study state is unspecified.
      ACTIVE: The study is active.
      INACTIVE: The study is stopped due to an internal error.
      COMPLETED: The study is done when the service exhausts the parameter
        search space or max_trial_count is reached.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        INACTIVE = 2
        COMPLETED = 3
    createTime = _messages.StringField(1)
    inactiveReason = _messages.StringField(2)
    name = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)
    studyConfig = _messages.MessageField('GoogleCloudMlV1StudyConfig', 5)