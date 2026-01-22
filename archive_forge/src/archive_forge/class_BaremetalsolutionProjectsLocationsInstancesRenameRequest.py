from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsInstancesRenameRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsInstancesRenameRequest object.

  Fields:
    name: Required. The `name` field is used to identify the instance. Format:
      projects/{project}/locations/{location}/instances/{instance}
    renameInstanceRequest: A RenameInstanceRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    renameInstanceRequest = _messages.MessageField('RenameInstanceRequest', 2)