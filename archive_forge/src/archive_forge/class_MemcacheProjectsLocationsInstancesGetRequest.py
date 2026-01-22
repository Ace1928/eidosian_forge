from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MemcacheProjectsLocationsInstancesGetRequest(_messages.Message):
    """A MemcacheProjectsLocationsInstancesGetRequest object.

  Fields:
    name: Required. Memcached instance resource name in the format:
      `projects/{project_id}/locations/{location_id}/instances/{instance_id}`
      where `location_id` refers to a GCP region
  """
    name = _messages.StringField(1, required=True)