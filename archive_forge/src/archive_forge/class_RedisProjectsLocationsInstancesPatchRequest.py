from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RedisProjectsLocationsInstancesPatchRequest(_messages.Message):
    """A RedisProjectsLocationsInstancesPatchRequest object.

  Fields:
    instance: A Instance resource to be passed as the request body.
    name: Required. Unique name of the resource in this scope including
      project and location using the form:
      `projects/{project_id}/locations/{location_id}/instances/{instance_id}`
      Note: Redis instances are managed and addressed at regional level so
      location_id here refers to a GCP region; however, users may choose which
      specific zone (or collection of zones for cross-zone instances) an
      instance should be provisioned in. Refer to location_id and
      alternative_location_id fields for more details.
    updateMask: Required. Mask of fields to update. At least one path must be
      supplied in this field. The elements of the repeated paths field may
      only include these fields from Instance: * `displayName` * `labels` *
      `memorySizeGb` * `redisConfig` * `replica_count`
  """
    instance = _messages.MessageField('Instance', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)