from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MemcacheProjectsLocationsInstancesRescheduleMaintenanceRequest(_messages.Message):
    """A MemcacheProjectsLocationsInstancesRescheduleMaintenanceRequest object.

  Fields:
    instance: Required. Memcache instance resource name using the form:
      `projects/{project_id}/locations/{location_id}/instances/{instance_id}`
      where `location_id` refers to a GCP region.
    rescheduleMaintenanceRequest: A RescheduleMaintenanceRequest resource to
      be passed as the request body.
  """
    instance = _messages.StringField(1, required=True)
    rescheduleMaintenanceRequest = _messages.MessageField('RescheduleMaintenanceRequest', 2)