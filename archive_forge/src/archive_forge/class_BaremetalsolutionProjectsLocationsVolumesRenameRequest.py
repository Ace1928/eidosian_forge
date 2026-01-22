from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsVolumesRenameRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsVolumesRenameRequest object.

  Fields:
    name: Required. The `name` field is used to identify the volume. Format:
      projects/{project}/locations/{location}/volumes/{volume}
    renameVolumeRequest: A RenameVolumeRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    renameVolumeRequest = _messages.MessageField('RenameVolumeRequest', 2)