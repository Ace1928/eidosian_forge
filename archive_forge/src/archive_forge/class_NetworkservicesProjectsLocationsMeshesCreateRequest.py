from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsMeshesCreateRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsMeshesCreateRequest object.

  Fields:
    mesh: A Mesh resource to be passed as the request body.
    meshId: Required. Short name of the Mesh resource to be created.
    parent: Required. The parent resource of the Mesh. Must be in the format
      `projects/*/locations/global`.
  """
    mesh = _messages.MessageField('Mesh', 1)
    meshId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)