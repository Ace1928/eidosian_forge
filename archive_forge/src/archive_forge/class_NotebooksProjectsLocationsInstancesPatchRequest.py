from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsInstancesPatchRequest(_messages.Message):
    """A NotebooksProjectsLocationsInstancesPatchRequest object.

  Fields:
    instance: A Instance resource to be passed as the request body.
    name: Output only. The name of this notebook instance. Format:
      `projects/{project_id}/locations/{location}/instances/{instance_id}`
    requestId: Optional. Idempotent request UUID.
    updateMask: Required. Mask used to update an instance
  """
    instance = _messages.MessageField('Instance', 1)
    name = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)
    updateMask = _messages.StringField(4)