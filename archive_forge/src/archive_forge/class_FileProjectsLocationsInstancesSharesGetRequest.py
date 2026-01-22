from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileProjectsLocationsInstancesSharesGetRequest(_messages.Message):
    """A FileProjectsLocationsInstancesSharesGetRequest object.

  Fields:
    name: Required. The share resource name, in the format `projects/{project_
      id}/locations/{location}/instances/{instance_id}/shares/{share_id}`
  """
    name = _messages.StringField(1, required=True)