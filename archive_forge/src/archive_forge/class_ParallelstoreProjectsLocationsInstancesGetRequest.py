from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ParallelstoreProjectsLocationsInstancesGetRequest(_messages.Message):
    """A ParallelstoreProjectsLocationsInstancesGetRequest object.

  Fields:
    name: Required. The instance resource name, in the format
      `projects/{project_id}/locations/{location}/instances/{instance_id}`.
  """
    name = _messages.StringField(1, required=True)