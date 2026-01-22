from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileProjectsLocationsInstancesDeleteRequest(_messages.Message):
    """A FileProjectsLocationsInstancesDeleteRequest object.

  Fields:
    force: If set to true, all snapshots of the instance will also be deleted.
      (Otherwise, the request will only work if the instance has no
      snapshots.)
    name: Required. The instance resource name, in the format
      `projects/{project_id}/locations/{location}/instances/{instance_id}`
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)