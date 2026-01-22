from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileProjectsLocationsInstancesRestoreRequest(_messages.Message):
    """A FileProjectsLocationsInstancesRestoreRequest object.

  Fields:
    name: Required. The resource name of the instance, in the format `projects
      /{project_number}/locations/{location_id}/instances/{instance_id}`.
    restoreInstanceRequest: A RestoreInstanceRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    restoreInstanceRequest = _messages.MessageField('RestoreInstanceRequest', 2)