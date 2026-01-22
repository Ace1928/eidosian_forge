from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsMeshesDeleteRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsMeshesDeleteRequest object.

  Fields:
    name: Required. A name of the Mesh to delete. Must be in the format
      `projects/*/locations/global/meshes/*`.
  """
    name = _messages.StringField(1, required=True)