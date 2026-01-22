from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1NasTrial(_messages.Message):
    """Represents a uCAIP NasJob trial.

  Enums:
    StateValueValuesEnum: Output only. The detailed state of the NasTrial.

  Fields:
    endTime: Output only. Time when the NasTrial's status changed to
      `SUCCEEDED` or `INFEASIBLE`.
    finalMeasurement: Output only. The final measurement containing the
      objective value.
    id: Output only. The identifier of the NasTrial assigned by the service.
    startTime: Output only. Time when the NasTrial was started.
    state: Output only. The detailed state of the NasTrial.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The detailed state of the NasTrial.

    Values:
      STATE_UNSPECIFIED: The NasTrial state is unspecified.
      REQUESTED: Indicates that a specific NasTrial has been requested, but it
        has not yet been suggested by the service.
      ACTIVE: Indicates that the NasTrial has been suggested.
      STOPPING: Indicates that the NasTrial should stop according to the
        service.
      SUCCEEDED: Indicates that the NasTrial is completed successfully.
      INFEASIBLE: Indicates that the NasTrial should not be attempted again.
        The service will set a NasTrial to INFEASIBLE when it's done but
        missing the final_measurement.
    """
        STATE_UNSPECIFIED = 0
        REQUESTED = 1
        ACTIVE = 2
        STOPPING = 3
        SUCCEEDED = 4
        INFEASIBLE = 5
    endTime = _messages.StringField(1)
    finalMeasurement = _messages.MessageField('GoogleCloudAiplatformV1Measurement', 2)
    id = _messages.StringField(3)
    startTime = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)