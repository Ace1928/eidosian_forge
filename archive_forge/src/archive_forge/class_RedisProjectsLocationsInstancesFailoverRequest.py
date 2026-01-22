from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RedisProjectsLocationsInstancesFailoverRequest(_messages.Message):
    """A RedisProjectsLocationsInstancesFailoverRequest object.

  Fields:
    failoverInstanceRequest: A FailoverInstanceRequest resource to be passed
      as the request body.
    name: Required. Redis instance resource name using the form:
      `projects/{project_id}/locations/{location_id}/instances/{instance_id}`
      where `location_id` refers to a GCP region.
  """
    failoverInstanceRequest = _messages.MessageField('FailoverInstanceRequest', 1)
    name = _messages.StringField(2, required=True)