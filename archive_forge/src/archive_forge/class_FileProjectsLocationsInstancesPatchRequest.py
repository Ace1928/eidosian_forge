from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileProjectsLocationsInstancesPatchRequest(_messages.Message):
    """A FileProjectsLocationsInstancesPatchRequest object.

  Fields:
    instance: A Instance resource to be passed as the request body.
    name: Output only. The resource name of the instance, in the format
      `projects/{project}/locations/{location}/instances/{instance}`.
    updateMask: Mask of fields to update. At least one path must be supplied
      in this field. The elements of the repeated paths field may only include
      these fields: * "description" * "file_shares" * "labels"
  """
    instance = _messages.MessageField('Instance', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)