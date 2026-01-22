from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsInstancesSetMachineTypeRequest(_messages.Message):
    """A NotebooksProjectsLocationsInstancesSetMachineTypeRequest object.

  Fields:
    name: Required. Format:
      `projects/{project_id}/locations/{location}/instances/{instance_id}`
    setInstanceMachineTypeRequest: A SetInstanceMachineTypeRequest resource to
      be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    setInstanceMachineTypeRequest = _messages.MessageField('SetInstanceMachineTypeRequest', 2)