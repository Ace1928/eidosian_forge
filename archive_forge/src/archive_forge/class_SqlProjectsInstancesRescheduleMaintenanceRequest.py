from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlProjectsInstancesRescheduleMaintenanceRequest(_messages.Message):
    """A SqlProjectsInstancesRescheduleMaintenanceRequest object.

  Fields:
    instance: Cloud SQL instance ID. This does not include the project ID.
    project: ID of the project that contains the instance.
    sqlInstancesRescheduleMaintenanceRequestBody: A
      SqlInstancesRescheduleMaintenanceRequestBody resource to be passed as
      the request body.
  """
    instance = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    sqlInstancesRescheduleMaintenanceRequestBody = _messages.MessageField('SqlInstancesRescheduleMaintenanceRequestBody', 3)