from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringServicesServiceLevelObjectivesCreateRequest(_messages.Message):
    """A MonitoringServicesServiceLevelObjectivesCreateRequest object.

  Fields:
    parent: Required. Resource name of the parent Service. The format is:
      projects/[PROJECT_ID_OR_NUMBER]/services/[SERVICE_ID]
    serviceLevelObjective: A ServiceLevelObjective resource to be passed as
      the request body.
    serviceLevelObjectiveId: Optional. The ServiceLevelObjective id to use for
      this ServiceLevelObjective. If omitted, an id will be generated instead.
      Must match the pattern ^[a-zA-Z0-9-_:.]+$
  """
    parent = _messages.StringField(1, required=True)
    serviceLevelObjective = _messages.MessageField('ServiceLevelObjective', 2)
    serviceLevelObjectiveId = _messages.StringField(3)