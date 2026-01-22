from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ReportInstanceStatusRequest(_messages.Message):
    """Request for ReportInstanceStatus.

  Fields:
    instanceUid: A unique ID for the instance which is guaranteed to be unique
      in case the user installs multiple hybrid runtimes with the same
      instance ID.
    reportTime: The time the report was generated in the runtime. Used to
      prevent an old status from overwriting a newer one. An instance should
      space out it's status reports so that clock skew does not play a factor.
    resources: Status for config resources
    spec: Resource spec.
  """
    instanceUid = _messages.StringField(1)
    reportTime = _messages.StringField(2)
    resources = _messages.MessageField('GoogleCloudApigeeV1ResourceStatus', 3, repeated=True)
    spec = _messages.MessageField('GoogleCloudApigeeV1ResourceSpec', 4)