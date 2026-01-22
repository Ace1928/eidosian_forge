from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringServicesServiceLevelObjectivesPatchRequest(_messages.Message):
    """A MonitoringServicesServiceLevelObjectivesPatchRequest object.

  Fields:
    name: Resource name for this ServiceLevelObjective. The format is: project
      s/[PROJECT_ID_OR_NUMBER]/services/[SERVICE_ID]/serviceLevelObjectives/[S
      LO_NAME]
    serviceLevelObjective: A ServiceLevelObjective resource to be passed as
      the request body.
    updateMask: A set of field paths defining which fields to use for the
      update.
  """
    name = _messages.StringField(1, required=True)
    serviceLevelObjective = _messages.MessageField('ServiceLevelObjective', 2)
    updateMask = _messages.StringField(3)